"""
detection/donnees.py
====================
Lecture du dataset d'entraînement (CSV, JSONL ou Parquet) et normalisation
des colonnes. Le fichier doit contenir au minimum : texte, label, split.

Les noms de colonnes acceptés sont définis dans config.COLONNES_DATASET.
La colonne `split` (train / val / test) est fournie par le dataset : elle est
construite par groupes (un document source et ses générations IA restent dans
le même split), ce qui évite les fuites de données entre entraînement et test.
"""

from pathlib import Path

import pandas as pd

from .. import config

COLONNES_STANDARD = ["texte", "label", "split", "domaine", "source", "generateur",
                     "longueur", "langue"]

_LABELS = {"ai": 1, "machine": 1, "generated": 1, "1": 1, "true": 1,
           "human": 0, "0": 0, "false": 0}
_CODES_LANGUE = {"en": {"en", "eng", "english", "anglais"},
                 "fr": {"fr", "fra", "fre", "french", "francais", "français"}}
_SPLITS = {"train": "train", "training": "train",
           "val": "val", "valid": "val", "validation": "val", "dev": "val",
           "test": "test"}


def _lire(chemin: Path) -> pd.DataFrame:
    ext = chemin.suffix.lower()
    if ext == ".parquet":
        import pyarrow.parquet as pq
        # Seules les colonnes utiles sont lues (économie de mémoire sur les gros fichiers)
        utiles = {nom for noms in config.COLONNES_DATASET.values() for nom in noms}
        colonnes = [c for c in pq.read_schema(chemin).names if c in utiles]
        return pd.read_parquet(chemin, columns=colonnes)
    if ext in (".jsonl", ".json"):
        return pd.read_json(chemin, lines=True)
    if ext == ".csv":
        return pd.read_csv(chemin)
    raise ValueError(f"Format non géré : {ext} (attendu : .parquet, .jsonl, .csv)")


def _normaliser_valeurs(serie: pd.Series, table: dict, nom: str) -> pd.Series:
    vus = serie.astype(str).str.strip().str.lower()
    inconnus = sorted(set(vus.unique()) - set(table))
    if inconnus:
        raise ValueError(f"Valeurs inconnues dans la colonne '{nom}' : {inconnus[:10]}")
    return vus.map(table).astype(int if nom == "label" else object)


def charger_dataset(chemin, langue: str | None = None) -> pd.DataFrame:
    """
    Charge le dataset et retourne un DataFrame aux colonnes standard :
    texte, label (1 = IA, 0 = humain), split (train/val/test), domaine, source,
    generateur, longueur, langue. Les colonnes optionnelles absentes valent "inconnu".

    Args:
        langue : "en" ou "fr" pour ne garder que les textes de cette langue, si le
                 dataset a une colonne de langue. Sans cette colonne, tous les textes
                 sont conservés (on suppose le fichier déjà dans la bonne langue).
    """
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Dataset introuvable : {chemin}")
    brut = _lire(chemin)

    sortie = pd.DataFrame(index=brut.index)
    obligatoires = ("texte", "label", "split")
    for role, candidats in config.COLONNES_DATASET.items():
        trouvee = next((c for c in candidats if c in brut.columns), None)
        if trouvee is None:
            if role in obligatoires:
                raise ValueError(
                    f"Colonne '{role}' introuvable (noms acceptés : {list(candidats)}). "
                    f"Colonnes présentes : {list(brut.columns)}. "
                    f"Ajoutez le bon nom dans config.COLONNES_DATASET.")
            sortie[role] = "inconnu"
        else:
            sortie[role] = brut[trouvee]

    sortie["texte"] = sortie["texte"].astype(str)
    sortie["label"] = _normaliser_valeurs(sortie["label"], _LABELS, "label")
    sortie["split"] = _normaliser_valeurs(sortie["split"], _SPLITS, "split")
    for col in ("domaine", "source", "generateur", "longueur", "langue"):
        sortie[col] = sortie[col].astype(str)

    if langue in _CODES_LANGUE:
        if (sortie["langue"] == "inconnu").all():
            print(f"Pas de colonne de langue : tous les textes sont supposés en '{langue}'.")
        else:
            garde = sortie["langue"].str.strip().str.lower().isin(_CODES_LANGUE[langue])
            print(f"Filtre langue '{langue}' : {int(garde.sum()):,} textes conservés "
                  f"sur {len(sortie):,}.")
            sortie = sortie[garde]
    return sortie[COLONNES_STANDARD].reset_index(drop=True)


def echantillon_stratifie(df: pd.DataFrame, n: int | None, seed: int = 42) -> pd.DataFrame:
    """
    Sous-échantillon d'environ n lignes, en conservant les proportions
    label × domaine. Retourne df tel quel si n est None ou >= len(df).
    """
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    frac = n / len(df)
    return (df.groupby(["label", "domaine"], group_keys=False)
              .sample(frac=frac, random_state=seed)
              .reset_index(drop=True))
