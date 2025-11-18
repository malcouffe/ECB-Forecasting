# Instructions d'Installation et d'Utilisation - Chronos-2

## 🚀 Installation Rapide

### Option 1 : Script automatique (recommandé)

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2 : Installation manuelle

#### 1. Créer un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate  # Sur macOS/Linux
```

#### 2. Installer les dépendances

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 📊 Utilisation

### 1. Activer l'environnement virtuel

```bash
source venv/bin/activate
```

### 2. Lancer Jupyter Notebook

```bash
jupyter notebook
```

### 3. Ouvrir le notebook

Dans l'interface Jupyter qui s'ouvre dans votre navigateur, cliquez sur `chronos2_exploration.ipynb`

## 📝 Contenu du Notebook

Le notebook `chronos2_exploration.ipynb` vous permet d'explorer:

1. **Configuration du modèle** - Architecture et paramètres
2. **Structure détaillée** - Encoder, decoder, embeddings
3. **Informations techniques** - Nombre de paramètres, taille en mémoire
4. **Fichiers du repository** - Contenu du modèle sur Hugging Face
5. **Analyse des composants** - Couches, modules, méthodes

## 🎯 À propos de Chronos-2

**Chronos-2** est développé par Amazon et supporte :

- ✅ Prévisions **univariées** (une seule série temporelle)
- ✅ Prévisions **multivariées** (plusieurs séries simultanément)
- ✅ Prévisions **avec covariables** (variables externes comme facteurs économiques)

## 📦 Dépendances Principales

- `chronos-forecasting>=2.0` - Package officiel Chronos
- `torch>=2.0.0` - Framework PyTorch
- `transformers>=4.35.0` - Hugging Face Transformers
- `jupyter>=1.0.0` - Environnement notebook

## 🔗 Ressources

- [Chronos-2 sur Hugging Face](https://huggingface.co/amazon/chronos-2)
- [Documentation Chronos](https://github.com/amazon-science/chronos-forecasting)
- [Paper Chronos](https://arxiv.org/abs/2403.07815)

## ⚠️ Notes Importantes

1. **Téléchargement du modèle** : La première fois que vous exécutez le notebook, le modèle sera téléchargé depuis Hugging Face (cela peut prendre quelques minutes selon votre connexion)

2. **Espace disque** : Assurez-vous d'avoir suffisamment d'espace disque pour le modèle (~1-2 GB)

3. **Mémoire RAM** : Le chargement du modèle complet nécessite au moins 4-8 GB de RAM

## 🛠️ Problèmes Courants

### Le modèle ne se télécharge pas
```bash
# Vérifiez votre connexion internet
# Réessayez en exécutant la cellule du notebook
```

### Erreur d'import
```bash
# Vérifiez que l'environnement virtuel est activé
source venv/bin/activate

# Réinstallez les dépendances
pip install -r requirements.txt --force-reinstall
```

## 💡 Astuce

Pour explorer uniquement la configuration sans télécharger le modèle complet, exécutez seulement les cellules 1 à 10 du notebook. Cela vous donnera déjà beaucoup d'informations sur l'architecture du modèle !

