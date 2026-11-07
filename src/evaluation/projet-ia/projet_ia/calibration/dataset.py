"""
calibration/dataset.py
======================
Lecture du meta-dataset (CSV) et préparation des features.
"""

import numpy as np
import pandas as pd


FEATURES_NUMERIQUES = [
    "nli",
    "similarite_ponderee",
    "sens_global",
    "penalite_neg",
    "penalite_entites",
    "penalite_concept",
]


def charger_dataset(chemin: str):
    """
    Charge le CSV et construit la matrice de features X (avec biais),
    le vecteur cible y, et la liste ordonnee des noms de colonnes de X
    (utile pour retrouver quel poids correspond a quelle feature).

    La colonne categorielle (ex: caractere_question) est encodee en
    one-hot dynamiquement : les categories trouvees dans le CSV deviennent
    chacune une colonne binaire "q_<categorie>".
    """
    df = pd.read_csv(chemin)

    colonnes_manquantes = [c for c in FEATURES_NUMERIQUES + ["label"] if c not in df.columns]
    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes dans {chemin} : {colonnes_manquantes}")

    col_cat = "caractere_question" if "caractere_question" in df.columns else None

    noms_colonnes = list(FEATURES_NUMERIQUES)
    blocs = [df[FEATURES_NUMERIQUES].to_numpy(dtype=float)]

    categories = []
    if col_cat:
        categories = sorted(df[col_cat].dropna().unique().tolist())
        for cat in categories:
            noms_colonnes.append(f"q_{cat}")
            blocs.append((df[col_cat] == cat).to_numpy(dtype=float).reshape(-1, 1))

    X = np.hstack(blocs)
    y = df["label"].to_numpy(dtype=float)

    # Validation : label doit etre binaire (0/1)
    valeurs_uniques = set(np.unique(y).tolist())
    if not valeurs_uniques.issubset({0.0, 1.0}):
        raise ValueError(f"La colonne 'label' doit etre binaire (0/1), trouve : {valeurs_uniques}")

    return X, y, noms_colonnes, categories


def normaliser_standard(X: np.ndarray):
    """
    Standardise chaque colonne (moyenne 0, ecart-type 1).
    Necessaire pour que la descente de gradient converge vite et de
    maniere stable quand les features ont des echelles tres differentes.
    Retourne X_norm, moyennes, ecarts_types (utiles pour re-appliquer
    la meme normalisation a de nouvelles donnees si besoin).
    """
    moyennes = X.mean(axis=0)
    ecarts   = X.std(axis=0)
    ecarts[ecarts == 0] = 1.0  # evite une division par zero (colonne constante)
    X_norm = (X - moyennes) / ecarts
    return X_norm, moyennes, ecarts
