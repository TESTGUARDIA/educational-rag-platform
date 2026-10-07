"""
detection/donnees.py
====================
Lecture du dataset d'entraînement (CSV, JSONL ou Parquet) et normalisation
des colonnes. Le fichier doit contenir au minimum : texte, label, split.

Les noms de colonnes acceptés sont définis dans config.COLONNES_DATASET.
Pour les gros fichiers Parquet, `charger_dataset(..., avec_texte=False)` ne lit que
les métadonnées et `iterer_textes()` lit les textes par lots : la mémoire ne dépend
plus de la taille du dataset.
La colonne `split` (train / val / test) est fournie par le dataset : elle est
construite par groupes (un document source et ses générations IA restent dans
le même split), ce qui évite les fuites de données entre entraînement et test.
"""

from collections.abc import Iterator
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


def _lire(chemin: Path, avec_texte: bool = True) -> pd.DataFrame:
    ext = chemin.suffix.lower()
    if ext == ".parquet":
        import pyarrow.parquet as pq
        # Seules les colonnes utiles sont lues (économie de mémoire sur les gros fichiers)
        roles = [r for r in config.COLONNES_DATASET if avec_texte or r != "texte"]
        utiles = {nom for r in roles for nom in config.COLONNES_DATASET[r]}
        colonnes = [c for c in pq.read_schema(chemin).names if c in utiles]
        return pd.read_parquet(chemin, columns=colonnes)
    if not avec_texte:
        raise ValueError("avec_texte=False n'est possible qu'avec un fichier .parquet")
    if ext in (".jsonl", ".json"):
        return pd.read_json(chemin, lines=True)
    if ext == ".csv":
        return pd.read_csv(chemin)
    raise ValueError(f"Format non géré : {ext} (attendu : .parquet, .jsonl, .csv)")


def _colonne(colonnes, role: str) -> str | None:
    """Nom de la colonne du fichier qui joue ce rôle (voir config.COLONNES_DATASET)."""
    return next((c for c in config.COLONNES_DATASET[role] if c in colonnes), None)


def _masque_langue(serie: pd.Series, langue: str | None):
    """Lignes de la langue demandée, ou None s'il n'y a pas à filtrer."""
    if langue not in _CODES_LANGUE or (serie == "inconnu").all():
        return None
    return serie.astype(str).str.strip().str.lower().isin(_CODES_LANGUE[langue]).to_numpy()


def _normaliser_valeurs(serie: pd.Series, table: dict, nom: str) -> pd.Series:
    vus = serie.astype(str).str.strip().str.lower()
    inconnus = sorted(set(vus.unique()) - set(table))
    if inconnus:
        raise ValueError(f"Valeurs inconnues dans la colonne '{nom}' : {inconnus[:10]}")
    return vus.map(table).astype(int if nom == "label" else object)


def charger_dataset(chemin, langue: str | None = None, avec_texte: bool = True) -> pd.DataFrame:
    """
    Charge le dataset et retourne un DataFrame aux colonnes standard :
    texte, label (1 = IA, 0 = humain), split (train/val/test), domaine, source,
    generateur, longueur, langue. Les colonnes optionnelles absentes valent "inconnu".

    Args:
        langue : "en" ou "fr" pour ne garder que les textes de cette langue, si le
                 dataset a une colonne de langue. Sans cette colonne, tous les textes
                 sont conservés (on suppose le fichier déjà dans la bonne langue).
        avec_texte : False pour ne pas lire les textes (Parquet seulement) : les
                     métadonnées seules, à compléter par iterer_textes().
    """
    chemin = Path(chemin)
    if not chemin.exists():
        raise FileNotFoundError(f"Dataset introuvable : {chemin}")
    brut = _lire(chemin, avec_texte)

    sortie = pd.DataFrame(index=brut.index)
    obligatoires = ("texte", "label", "split") if avec_texte else ("label", "split")
    for role, candidats in config.COLONNES_DATASET.items():
        if role == "texte" and not avec_texte:
            continue
        trouvee = _colonne(brut.columns, role)
        if trouvee is None:
            if role in obligatoires:
                raise ValueError(
                    f"Colonne '{role}' introuvable (noms acceptés : {list(candidats)}). "
                    f"Colonnes présentes : {list(brut.columns)}. "
                    f"Ajoutez le bon nom dans config.COLONNES_DATASET.")
            sortie[role] = "inconnu"
        else:
            sortie[role] = brut[trouvee]

    if avec_texte:
        sortie["texte"] = sortie["texte"].astype(str)
    sortie["label"] = _normaliser_valeurs(sortie["label"], _LABELS, "label")
    sortie["split"] = _normaliser_valeurs(sortie["split"], _SPLITS, "split")
    for col in ("domaine", "source", "generateur", "longueur", "langue"):
        sortie[col] = sortie[col].astype(str)

    if langue in _CODES_LANGUE:
        garde = _masque_langue(sortie["langue"], langue)
        if garde is None:
            print(f"Pas de colonne de langue : tous les textes sont supposés en '{langue}'.")
        else:
            print(f"Filtre langue '{langue}' : {int(garde.sum()):,} textes conservés "
                  f"sur {len(sortie):,}.")
            sortie = sortie[garde]
    colonnes = [c for c in COLONNES_STANDARD if avec_texte or c != "texte"]
    return sortie[colonnes].reset_index(drop=True)


def iterer_textes(chemin, split: str, langue: str | None = None,
                  taille_lot: int = 10_000, lignes_par_lecture: int = 50_000) -> Iterator[list[str]]:
    """
    Textes d'un split d'un fichier Parquet, dans l'ordre du fichier (donc dans
    l'ordre de charger_dataset), par lots de `taille_lot`. Le fichier est lu par
    morceaux : seuls quelques dizaines de milliers de textes sont en mémoire.
    """
    import pyarrow.parquet as pq

    fichier = pq.ParquetFile(chemin)
    noms = fichier.schema_arrow.names
    col_texte, col_split = _colonne(noms, "texte"), _colonne(noms, "split")
    col_langue = _colonne(noms, "langue")
    colonnes = [c for c in (col_texte, col_split, col_langue) if c]
    tampon = []
    for lot in fichier.iter_batches(batch_size=lignes_par_lecture, columns=colonnes):
        df = lot.to_pandas()
        garde = (_normaliser_valeurs(df[col_split], _SPLITS, "split") == split).to_numpy()
        if col_langue:
            masque = _masque_langue(df[col_langue], langue)
            if masque is not None:
                garde = garde & masque          # (pas &= : tableau Arrow en lecture seule)
        tampon.extend(df[col_texte][garde].astype(str).tolist())
        while len(tampon) >= taille_lot:
            yield tampon[:taille_lot]
            tampon = tampon[taille_lot:]
    if tampon:
        yield tampon


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
