# Exploration du modèle Chronos-2 pour les prévisions ECB

Ce projet permet d'explorer le modèle **Chronos-2** d'Amazon pour la prévision de séries temporelles.

## 🎯 À propos de Chronos-2

**Chronos-2** est un modèle de fondation pour la prévision de séries temporelles développé par Amazon Research. Il supporte :

- ✅ **Prévisions univariées** - Analyse d'une seule série temporelle
- ✅ **Prévisions multivariées** - Analyse simultanée de plusieurs séries
- ✅ **Prévisions avec covariables** - Intégration de variables externes

Tout cela dans une **architecture unique et unifiée** basée sur les Transformers !

## 🚀 Installation Rapide

### Option 1 : Script automatique (recommandé)

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2 : Installation manuelle

```bash
# 1. Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 2. Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Utilisation

```bash
# 1. Activer l'environnement
source venv/bin/activate

# 2. Lancer Jupyter Notebook
jupyter notebook

# 3. Ouvrir chronos2_exploration.ipynb dans votre navigateur
```

## 📝 Contenu du Projet

- `chronos2_exploration.ipynb` - Notebook interactif pour explorer le modèle
- `requirements.txt` - Dépendances Python nécessaires
- `setup.sh` - Script d'installation automatique
- `INSTRUCTIONS.md` - Guide détaillé d'utilisation

## 📚 Ce que vous allez découvrir

Le notebook vous permet d'explorer :

1. 🔍 **Architecture du modèle** - Structure complète et configuration
2. 📊 **Paramètres** - Nombre total, taille en mémoire
3. 🏗️ **Composants** - Encoder, decoder, embeddings
4. 📦 **Fichiers** - Contenu du repository Hugging Face
5. 🛠️ **Méthodes** - Fonctions disponibles du modèle

## 🔗 Ressources

- [Chronos-2 sur Hugging Face](https://huggingface.co/amazon/chronos-2)
- [GitHub - Chronos Forecasting](https://github.com/amazon-science/chronos-forecasting)
- [Paper Chronos](https://arxiv.org/abs/2403.07815)

## 💡 Astuce

Vous pouvez explorer la configuration du modèle sans le télécharger complètement ! Exécutez seulement les cellules 1 à 10 du notebook pour obtenir toutes les informations d'architecture sans télécharger les poids complets.

---

**Note** : Pour des instructions détaillées, consultez le fichier `INSTRUCTIONS.md`

