#!/bin/bash
# Script d'installation pour le projet Chronos-2

echo "=========================================="
echo "Configuration de l'environnement Chronos-2"
echo "=========================================="
echo ""

# Créer l'environnement virtuel
echo "1. Création de l'environnement virtuel..."
python3 -m venv venv
echo "✓ Environnement virtuel créé"
echo ""

# Activer l'environnement
echo "2. Activation de l'environnement..."
source venv/bin/activate
echo "✓ Environnement activé"
echo ""

# Mettre à jour pip
echo "3. Mise à jour de pip..."
pip install --upgrade pip
echo "✓ pip mis à jour"
echo ""

# Installer les dépendances
echo "4. Installation des dépendances..."
pip install -r requirements.txt
echo "✓ Dépendances installées"
echo ""

echo "=========================================="
echo "Installation terminée avec succès !"
echo "=========================================="
echo ""
echo "Pour activer l'environnement:"
echo "  source venv/bin/activate"
echo ""
echo "Pour lancer Jupyter Notebook:"
echo "  jupyter notebook"
echo ""
echo "Puis ouvrez: chronos2_exploration.ipynb"
echo ""

