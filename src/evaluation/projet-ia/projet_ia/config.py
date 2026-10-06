"""
config.py
=========
Configuration centrale du projet : chemins des fichiers et noms des modèles.

Tout ce qui est "réglable" (où sont les datasets, quel modèle spaCy utiliser…)
se trouve ici, pour ne pas avoir de chemins ou de noms de modèles écrits en dur
dans le reste du code.

Le dossier racine du projet peut être surchargé avec la variable d'environnement
PROJET_IA_RACINE (utile si le paquet est installé hors du dépôt).
"""

import os
from pathlib import Path

# ── Dossiers ──────────────────────────────────────────────────────────────────
RACINE = Path(os.environ.get("PROJET_IA_RACINE", Path(__file__).resolve().parent.parent))
DOSSIER_DATA   = RACINE / "data"     # datasets (CSV) en entrée
DOSSIER_MODELS = RACINE / "models"   # fichiers générés : poids, génomes, modèles .pkl

# ── Modèles de langage ────────────────────────────────────────────────────────
MODELE_CAMEMBERT = "camembert-base"

# Vérification de réponses : fr_core_news_md (vecteurs GloVe nécessaires à token.similarity())
SPACY_VERIFICATION = "fr_core_news_md"

# Vérification : un seul modèle NLI, volontairement sans repli (les poids calibrés
# dépendent de ce modèle précis).
MODELES_NLI_VERIFICATION = ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0",)
# Détection : on prend le premier modèle disponible dans cette liste.
MODELES_NLI_DETECTION = (
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    "facebook/bart-large-mnli",
)

# ── Fichiers : vérification / calibration ─────────────────────────────────────
CHEMIN_META_DATASET   = DOSSIER_DATA / "meta_dataset.csv"
CHEMIN_POIDS_CALIBRES = DOSSIER_MODELS / "poids_calibres.json"

# ── Détection IA : langues ────────────────────────────────────────────────────
# Chaque langue a son encodeur, son modèle spaCy (outils d'analyse grammaticale) et
# ses propres fichiers (cache et modèles dans un sous-dossier au nom de la langue).
#   "en"    : anglais       "fr" : français
#   "multi" : un seul modèle entraîné sur des textes mixtes (encodeur multilingue)
LANGUES = {
    "en":    {"nom": "anglais",      "encodeur": "roberta-base",     "spacy": "en_core_web_sm"},
    "fr":    {"nom": "français",     "encodeur": "camembert-base",   "spacy": "fr_core_news_sm"},
    "multi": {"nom": "multilingue",  "encodeur": "xlm-roberta-base", "spacy": None},
}
LANGUE_PAR_DEFAUT = "en"          # le dataset d'entraînement actuel est en anglais
MAX_TOKENS_DETECTION = 128        # troncature PAR PHRASE (les phrases sont courtes)
POOLING_DETECTION    = "mean"     # "mean" (moyenne des tokens) ou "cls"
MAX_PHRASES = 30                  # phrases analysées au maximum par paragraphe
PHRASES_PAR_TEXTE_TRAIN = 3       # phrases conservées par paragraphe pour entraîner le modèle-phrase


def verifier_langue(langue: str) -> str:
    if langue not in LANGUES:
        raise ValueError(f"Langue inconnue : '{langue}'. Choix : {list(LANGUES)}")
    return langue


class CheminsDetection:
    """Chemins des fichiers de détection pour une langue donnée."""

    def __init__(self, langue: str = LANGUE_PAR_DEFAUT):
        self.langue = verifier_langue(langue)
        self.cache = DOSSIER_DATA / "cache" / langue
        self.dossier_modeles = DOSSIER_MODELS / langue
        self.modele = self.dossier_modeles / "modele_detection.pkl"
        self.rapport = self.dossier_modeles / "rapport_entrainement.json"

    def genome(self, composant: str):
        return self.dossier_modeles / f"genome_{composant}.json"

    def historique(self, composant: str):
        return self.dossier_modeles / f"historique_evolution_{composant}.json"


# ── Détection IA : fichier du dataset ─────────────────────────────────────────
CHEMIN_DATASET_DETECTION = DOSSIER_DATA / "dataset.parquet"

# ── Détection IA : colonnes du dataset ────────────────────────────────────────
# Pour chaque rôle, les noms de colonnes acceptés (le premier trouvé est utilisé).
# Si votre fichier utilise d'autres noms, ajoutez-les ici.
COLONNES_DATASET = {
    "texte":      ("text", "texte", "content"),
    "label":      ("label", "is_ai", "generated"),
    "split":      ("split",),
    "domaine":    ("domain", "domaine"),
    "source":     ("source",),
    "generateur": ("model", "generator", "generateur"),
    "longueur":   ("length_category", "length", "longueur"),
    "langue":     ("lang", "language", "langue"),
}
