#!/usr/bin/env python3
"""
LoRA fine-tuning scaffold for Moirai-2 on the ECB investment dataset.

The script uses the helpers defined in this repository to:
* load the stacked quarterly dataset
* construct sliding windows for all (or selected) countries
* attach lightweight LoRA adapters on top of the Moirai-2 linear projections
* optimize the adapters with a simple quantile (pinball) loss

Only the LoRA weights are saved to disk so the original checkpoint can remain
frozen and re-used elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ecb_forecast.compare import resolve_device
from ecb_forecast.datasets import load_quarterly_dataset
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
from uni2ts.transform.imputation import CausalMeanImputation


@dataclass(frozen=True)
class Window:
    past_target: np.ndarray
    past_observed: np.ndarray
    future_target: np.ndarray
    future_observed: np.ndarray
    series_id: str


class SlidingWindowDataset(Dataset):
    """Materialize sliding windows across all series for supervised training."""

    def __init__(
        self,
        dataframe,
        context_length: int,
        prediction_length: int,
        stride: int,
        id_column: str,
        target_column: str,
        allowed_series_ids: set[str] | None = None,
    ):
        if context_length <= 0 or prediction_length <= 0:
            raise ValueError("Context length and prediction length must be > 0.")
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.windows: list[Window] = []
        self._build_windows(
            dataframe,
            context_length,
            prediction_length,
            stride,
            id_column,
            target_column,
            allowed_series_ids,
        )

    def _build_windows(
        self,
        dataframe,
        context_length: int,
        prediction_length: int,
        stride: int,
        id_column: str,
        target_column: str,
        allowed_series_ids: set[str] | None,
    ) -> None:
        stride = max(1, stride)
        imputer = CausalMeanImputation()
        for series_id, group in dataframe.groupby(id_column):
            normalized_id = str(series_id)
            if allowed_series_ids and normalized_id not in allowed_series_ids:
                continue
            values = group.sort_values("timestamp")[target_column].to_numpy(dtype=np.float32)
            observed = ~np.isnan(values)
            if len(values) < context_length + prediction_length:
                continue
            # Use causal mean imputation to preserve the training protocol.
            imputed = imputer(values[:, None]).squeeze(-1)
            for start in range(
                0, len(values) - context_length - prediction_length + 1, stride
            ):
                ctx = imputed[start : start + context_length]
                ctx_obs = observed[start : start + context_length]
                fut = imputed[start + context_length : start + context_length + prediction_length]
                fut_obs = observed[
                    start + context_length : start + context_length + prediction_length
                ]
                if not ctx_obs.any() or not fut_obs.any():
                    continue
                self.windows.append(
                    Window(
                        past_target=ctx.copy(),
                        past_observed=ctx_obs.copy(),
                        future_target=fut.copy(),
                        future_observed=fut_obs.copy(),
                        series_id=normalized_id,
                    )
                )

    def __len__(self) -> int:  # noqa: D401 - standard Dataset contract
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        window = self.windows[index]
        past_target = torch.from_numpy(window.past_target).float().unsqueeze(-1)
        past_observed = torch.from_numpy(window.past_observed).bool().unsqueeze(-1)
        future_target = torch.from_numpy(window.future_target).float()
        future_observed = torch.from_numpy(window.future_observed).bool()
        return {
            "past_target": past_target,
            "past_observed_target": past_observed,
            "past_is_pad": torch.zeros(past_target.shape[0], dtype=torch.bool),
            "future_target": future_target,
            "future_observed": future_observed,
        }


class LoRALinear(nn.Module):
    """Simple LoRA adapter on top of an existing Linear projection."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0.")
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base_layer.out_features, bias=False)
        self.reset_parameters()
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: D401 - linear layer contract
        result = self.base_layer(input)
        lora_update = self.lora_b(self.lora_a(self.dropout(input))) * self.scaling
        return result + lora_update


def attach_lora_adapters(
    module: nn.Module,
    target_suffixes: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> list[str]:
    """Replace selected Linear layers with LoRA-enhanced versions."""

    def matches_target(name: str) -> bool:
        return any(name.endswith(suffix) for suffix in target_suffixes)

    replaced: list[str] = []

    def recurse(current: nn.Module, prefix: str = "") -> None:
        for name, child in current.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and matches_target(full_name):
                setattr(current, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced.append(full_name)
            else:
                recurse(child, full_name)

    recurse(module)
    if not replaced:
        raise RuntimeError(
            "LoRA injection did not match any modules. "
            "Check the --lora-targets argument."
        )
    return replaced


def freeze_base_parameters(module: nn.Module) -> None:
    for name, param in module.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


def pinball_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Standard quantile (pinball) loss with masking for missing targets."""
    target = target.unsqueeze(1)
    mask = mask.unsqueeze(1).float()
    diff = target - predictions
    quantiles = quantiles.view(1, -1, 1)
    loss = torch.maximum(quantiles * diff, (quantiles - 1) * diff)
    weighted_loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted_loss.sum() / denom


def split_dataset(
    dataset: Dataset,
    val_ratio: float,
    seed: int,
) -> tuple[Subset, Subset]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("Validation ratio must be between 0 and 1.")
    num_items = len(dataset)
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    split = int(num_items * (1 - val_ratio))
    train_indices = indices[:split]
    val_indices = indices[split:]
    if not train_indices or not val_indices:
        raise RuntimeError("Not enough samples to perform the requested split.")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def create_dataloaders(
    dataset: Dataset,
    batch_size: int,
    val_ratio: float,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_set, val_set = split_dataset(dataset, val_ratio, seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def save_lora_weights(
    module: nn.Module,
    output_dir: Path,
    metadata: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_state = {
        name: tensor.cpu()
        for name, tensor in module.state_dict().items()
        if "lora_" in name
    }
    artifact = {
        "metadata": metadata,
        "state_dict": lora_state,
    }
    output_path = output_dir / "moirai2_lora_adapter.pt"
    torch.save(artifact, output_path)
    with open(output_dir / "moirai2_lora_adapter.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return output_path


def run_epoch(
    loader: DataLoader,
    model: Moirai2Forecast,
    quantiles: torch.Tensor,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(mode=is_training)
    model.module.train(mode=is_training)
    total_loss = 0.0
    num_batches = 0
    for batch in loader:
        past_target = batch["past_target"].to(device)
        past_observed = batch["past_observed_target"].to(device)
        past_is_pad = batch["past_is_pad"].to(device)
        future_target = batch["future_target"].to(device)
        future_observed = batch["future_observed"].to(device)
        preds = model(
            past_target=past_target,
            past_observed_target=past_observed,
            past_is_pad=past_is_pad,
        )
        loss = pinball_loss(preds, future_target, future_observed, quantiles)
        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().item()
        num_batches += 1
    return total_loss / max(1, num_batches)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Moirai-2 with LoRA adapters on the ECB investment dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/investment_quarterly.csv"),
        help="Path to the stacked quarterly CSV.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Salesforce/moirai-2.0-R-small",
        help="Name (or local path) of the pre-trained checkpoint on Hugging Face Hub.",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=40,
        help="Number of quarters used as conditioning signal.",
    )
    parser.add_argument(
        "--prediction-length",
        type=int,
        default=4,
        help="Number of quarters to forecast during training.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Sliding window stride when generating training samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for both training and validation loaders.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Adapter learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay applied to adapter parameters.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, mps).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for dataset shuffling.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of windows reserved for validation.",
    )
    parser.add_argument(
        "--series-id",
        type=str,
        nargs="*",
        help="Optional subset of country codes to train on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("lora_artifacts"),
        help="Directory where the LoRA adapter weights will be stored.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="Rank of the LoRA adapters.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="Scaling factor applied to the LoRA update.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="Dropout applied to the LoRA branch.",
    )
    parser.add_argument(
        "--lora-targets",
        type=str,
        default="q_proj,k_proj,v_proj,out_proj,fc1,fc2,fc_gate",
        help="Comma-separated list of module suffixes to wrap with LoRA adapters.",
    )
    return parser.parse_args(argv)


def load_moirai_module(model_name: str, map_location: str) -> Moirai2Module:
    """Load a Moirai-2 checkpoint and surface a helpful error if the config is missing."""

    try:
        return Moirai2Module.from_pretrained(model_name, map_location=map_location)
    except TypeError as exc:
        message = str(exc)
        expected = ("d_model", "d_ff", "num_layers", "patch_size", "max_seq_len")
        if "missing" in message and all(token in message for token in expected):
            raise RuntimeError(
                f"Failed to load '{model_name}'. The checkpoint snapshot is missing its "
                "config.json so the Moirai-2 module cannot be instantiated. "
                "Download the full checkpoint (e.g. with 'huggingface-cli download "
                f"{model_name} --local-dir <path>') and point --model-name to that directory, "
                "or use a model snapshot that is already cached locally such as "
                "'Salesforce/moirai-2.0-R-small'."
            ) from exc
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    device_str = resolve_device(args.device)
    if device_str == "mps":
        print(
            "MPS device requested but Moirai fine-tuning still relies on operations that are "
            "unimplemented on MPS. Falling back to CPU; pass '--device cpu' to silence this message."
        )
        device_str = "cpu"
    device = torch.device(device_str)

    dataframe, summary = load_quarterly_dataset(args.dataset)
    print(
        f"Loaded quarterly dataset with {summary.num_rows} rows spanning "
        f"{len(summary.series_ids)} countries ({summary.start.date()} -> {summary.end.date()})."
    )
    allowed_ids = set(args.series_id) if args.series_id else None

    dataset = SlidingWindowDataset(
        dataframe=dataframe,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.stride,
        id_column="country",
        target_column="investment",
        allowed_series_ids=allowed_ids,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training windows were generated. Adjust the hyper-parameters.")
    print(f"Prepared {len(dataset)} windows ({args.context_length}+{args.prediction_length}).")

    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(
        f"Training batches: {len(train_loader)} | Validation batches: {len(val_loader)} "
        f"| Batch size: {args.batch_size}"
    )

    module = load_moirai_module(args.model_name, map_location="cpu")
    lora_targets = [token.strip() for token in args.lora_targets.split(",") if token.strip()]
    replaced = attach_lora_adapters(
        module,
        target_suffixes=lora_targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    freeze_base_parameters(module)
    print(f"Attached LoRA adapters to {len(replaced)} projections.")

    forecast = Moirai2Forecast(
        prediction_length=args.prediction_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=args.context_length,
        module=module,
    )
    forecast.to(device)

    quantiles = torch.tensor(module.quantile_levels, device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        (param for param in forecast.parameters() if param.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(train_loader, forecast, quantiles, device, optimizer=optimizer)
        val_loss = run_epoch(val_loader, forecast, quantiles, device, optimizer=None)
        print(f"Epoch {epoch:02d} | Train pinball loss: {train_loss:.4f} | Val pinball loss: {val_loss:.4f}")
        best_val = min(best_val, val_loss)

    metadata = {
        "model_name": args.model_name,
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_targets": lora_targets,
        "device": device_str,
        "best_val_pinball": best_val,
        "num_windows": len(dataset),
    }
    artifact_path = save_lora_weights(module, args.output_dir, metadata)
    print(f"Saved LoRA adapter to {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
