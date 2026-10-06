"""
detection/vectorisation.py
==========================
Transforme des phrases en lignes de features (stats + embedding CamemBERT)
et construit le dataset CSV vectorisé.
"""

import json

import pandas as pd

from ..common.embeddings import embedding_cls
from .grammaire import analyser_phrase, calculer_statistiques


def embedding_phrase(phrase: str) -> list[float]:
    """Calcule l'embedding CamemBERT d'une phrase (vecteur [CLS], 768 dims)."""
    return embedding_cls(phrase).tolist()


def vectoriser_phrase(phrase: str, label: bool) -> dict:
    """
    Transforme une phrase en une ligne de features pour le CSV d'entraînement.
    Colonnes : phrase, total_mots, total_ponctuation, embedding (JSON), label.
    """
    analyse = analyser_phrase(phrase)
    stats   = calculer_statistiques(analyse)
    vecteur = embedding_phrase(phrase)

    return {
        "phrase":            phrase,
        "total_mots":        stats["total_mots"],
        "total_ponctuation": stats["total_ponctuation"],
        "embedding":         json.dumps(vecteur),
        "label":             1 if label else 0,
    }


def construire_dataset(phrases: list[tuple], chemin_csv: str = "dataset.csv") -> pd.DataFrame:
    """
    Vectorise un corpus de phrases et exporte le résultat en CSV.

    Args:
        phrases    : liste de tuples (phrase: str, label: bool)
                     label=True → écrit par IA, label=False → écrit par un humain
        chemin_csv : chemin du fichier CSV de sortie

    Returns:
        DataFrame avec 5 colonnes : phrase, total_mots, total_ponctuation, embedding, label
    """
    lignes = []
    for i, (phrase, label) in enumerate(phrases):
        print(f"  [{i+1}/{len(phrases)}] {phrase[:60]}")
        lignes.append(vectoriser_phrase(phrase, label))

    df = pd.DataFrame(lignes)
    df.to_csv(chemin_csv, index=False, encoding="utf-8")
    print(f"\n✅ Dataset exporté : {chemin_csv}  ({len(df)} lignes × {len(df.columns)} colonnes)")
    return df
