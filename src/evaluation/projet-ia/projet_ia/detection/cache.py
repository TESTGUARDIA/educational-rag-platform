"""
detection/cache.py
==================
Pré-calcul UNIQUE des features de tout le dataset, en une seule passe dans l'encodeur.

Chaque paragraphe est découpé en phrases, chaque phrase est encodée UNE fois, puis
on en déduit tout le reste sans nouvel encodage :
  * niveau PARAGRAPHE : embedding moyen des phrases, features de style, cohérence
  * niveau PHRASE     : embedding + features de chaque phrase (pour le modèle-phrase)
Le paragraphe passe aussi UNE fois dans un modèle de langue causal pour les features
de perplexité (voir perplexite.py), aux deux niveaux.

Le calcul se fait par blocs ("chunks") enregistrés au fur et à mesure : si le calcul
est interrompu (des heures sur un gros dataset), relancer la même commande reprend
là où il s'était arrêté.

Contenu du dossier de cache (un dossier par langue) :
    cache_info.json            paramètres (langue, encodeur, …)
    meta_<split>.csv           label, domaine, source, generateur, longueur
    <split>_00000.npz …        emb_moy, style, coh, ppl (par paragraphe)
                               ph_emb, ph_style, ph_ppl, ph_idx (par phrase ; ph_idx = n° du
                               paragraphe)

Pour le split train, seules `phrases_par_texte_train` phrases tirées au hasard par
paragraphe sont conservées au niveau phrase (économie de disque et de mémoire) ;
val et test conservent toutes les phrases, pour évaluer comme en conditions réelles.
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..common.embeddings import encoder_textes
from .donnees import echantillon_stratifie
from .perplexite import N_FEATURES_PPL, N_FEATURES_PPL_PHRASE, calculer_tokens, features_perplexite
from .style import (N_FEATURES_COHERENCE, N_FEATURES_PHRASE, calculer_style,
                    coherence_depuis_embeddings, decouper_phrases, features_phrase)

VERSION_CACHE = 3
SPLITS = ("train", "val", "test")
COLONNES_META = ["label", "domaine", "source", "generateur", "longueur"]


def encodeur_depuis_info(info: dict):
    """Reconstruit la fonction d'encodage correspondant aux paramètres d'un cache."""
    return partial(encoder_textes, nom_modele=info["encodeur"],
                   max_length=info["max_tokens"], pooling=info["pooling"])


def calculateur_ppl_depuis_info(info: dict):
    """Reconstruit la fonction de perplexité (textes -> stats par token) d'un cache."""
    return partial(calculer_tokens, nom_modele=info["modele_perplexite"],
                   max_length=info["max_tokens_perplexite"])


def lire_info_cache(dossier) -> dict:
    chemin = Path(dossier) / "cache_info.json"
    if not chemin.exists():
        raise FileNotFoundError(
            f"Aucun cache dans {dossier}. Lancez d'abord : python -m projet_ia detection preparer")
    info = json.loads(chemin.read_text(encoding="utf-8"))
    if info.get("version") != VERSION_CACHE:
        raise ValueError(f"Cache {dossier} d'une ancienne version : supprimez-le et relancez "
                         f"'detection preparer'.")
    return info


def decouper_pour_cache(texte: str, max_phrases: int) -> list[str]:
    """Phrases d'un texte, limitées à max_phrases ; le texte entier s'il n'y en a aucune."""
    return decouper_phrases(texte)[:max_phrases] or [texte]


def construire_cache(df: pd.DataFrame, dossier=None,
                     langue: str = config.LANGUE_PAR_DEFAUT, nom_encodeur: str | None = None,
                     taille_chunk: int = 10_000, max_par_split: dict | None = None,
                     n_jobs: int = 1, max_tokens: int = config.MAX_TOKENS_DETECTION,
                     pooling: str = config.POOLING_DETECTION,
                     max_phrases: int = config.MAX_PHRASES,
                     phrases_par_texte_train: int | None = config.PHRASES_PAR_TEXTE_TRAIN,
                     nom_modele_ppl: str | None = None,
                     max_tokens_ppl: int = config.MAX_TOKENS_PERPLEXITE,
                     encodeur=None, calculateur_ppl=None, seed: int = 42) -> dict:
    """
    Calcule et enregistre les features de chaque split du dataset (voir donnees.charger_dataset).

    Args:
        langue         : "en", "fr" ou "multi" (choisit l'encodeur et le dossier par défaut)
        max_par_split  : ex. {"train": 100_000, "val": 20_000, "test": 20_000} pour ne
                         traiter qu'un sous-échantillon stratifié (essais rapides).
        encodeur       : fonction textes -> embeddings. Par défaut, l'encodeur Transformers
                         de la langue. (Permet d'injecter un faux encodeur en test.)
        nom_modele_ppl : modèle de langue causal de la perplexité (défaut : celui de la langue)
        calculateur_ppl: fonction textes -> stats par token (voir perplexite.calculer_tokens) ;
                         injectable en test, comme `encodeur`.
    Returns:
        dict info du cache
    """
    config.verifier_langue(langue)
    dossier = Path(dossier or config.CheminsDetection(langue).cache)
    dossier.mkdir(parents=True, exist_ok=True)
    nom_encodeur = nom_encodeur or config.LANGUES[langue]["encodeur"]
    nom_modele_ppl = nom_modele_ppl or config.LANGUES[langue]["perplexite"]
    max_par_split = max_par_split or {}

    info = {"version": VERSION_CACHE, "langue": langue, "encodeur": nom_encodeur,
            "max_tokens": max_tokens, "pooling": pooling, "max_phrases": max_phrases,
            "phrases_par_texte_train": phrases_par_texte_train, "seed": seed,
            "modele_perplexite": nom_modele_ppl, "max_tokens_perplexite": max_tokens_ppl,
            "max_par_split": {k: v for k, v in max_par_split.items() if v}}
    chemin_info = dossier / "cache_info.json"
    if chemin_info.exists():
        existant = json.loads(chemin_info.read_text(encoding="utf-8"))
        if existant != info:
            raise ValueError(
                f"Le cache de {dossier} a été construit avec d'autres paramètres :\n"
                f"  existant : {existant}\n  demandé  : {info}\n"
                f"Supprimez le dossier ou choisissez-en un autre (--cache).")
    chemin_info.write_text(json.dumps(info, indent=2), encoding="utf-8")

    if encodeur is None:
        encodeur = encodeur_depuis_info(info)
    if calculateur_ppl is None:
        calculateur_ppl = calculateur_ppl_depuis_info(info)

    for num_split, split in enumerate(SPLITS):
        sous = df[df["split"] == split]
        if len(sous) == 0:
            print(f"[{split}] aucun exemple — ignoré")
            continue
        sous = echantillon_stratifie(sous, max_par_split.get(split), seed)
        sous[COLONNES_META].to_csv(dossier / f"meta_{split}.csv", index=False)

        n_chunks = (len(sous) + taille_chunk - 1) // taille_chunk
        for k in range(n_chunks):
            fichier = dossier / f"{split}_{k:05d}.npz"
            if fichier.exists():
                continue                                    # reprise : déjà calculé
            textes = sous["texte"].iloc[k * taille_chunk:(k + 1) * taille_chunk].tolist()
            rng = np.random.default_rng([seed, num_split, k])
            contenu = _calculer_chunk(
                textes, k * taille_chunk, encodeur, calculateur_ppl, n_jobs, max_phrases,
                phrases_par_texte_train if split == "train" else None, rng)
            tmp = dossier / f"{split}_{k:05d}.tmp.npz"
            np.savez(tmp, **contenu)
            tmp.rename(fichier)                             # écriture atomique
            print(f"[{split}] bloc {k + 1}/{n_chunks} enregistré")
    return info


def _calculer_chunk(textes, decalage, encodeur, calculateur_ppl, n_jobs, max_phrases,
                    phrases_gardees, rng) -> dict:
    """Features d'un bloc de paragraphes (une seule passe dans l'encodeur et dans le
    modèle de langue)."""
    phrases = [decouper_pour_cache(t, max_phrases) for t in textes]
    plat = [p for ph in phrases for p in ph]
    emb = np.asarray(encodeur(plat), dtype=np.float32)
    tokens = calculateur_ppl(textes)

    emb_moy, coh, ppl = [], [], []
    ph_emb, ph_style, ph_ppl, ph_idx = [], [], [], []
    debut = 0
    for i, ph in enumerate(phrases):
        n = len(ph)
        e = emb[debut:debut + n]
        emb_moy.append(e.mean(axis=0))
        coh.append(coherence_depuis_embeddings(e))
        ppl_texte, ppl_phrases = features_perplexite(textes[i], ph, tokens[i])
        ppl.append(ppl_texte)

        garder = np.arange(n)
        if phrases_gardees and n > phrases_gardees:         # train : sous-échantillon de phrases
            garder = np.sort(rng.choice(n, phrases_gardees, replace=False))
        for j in garder:
            ph_emb.append(e[j])
            ph_style.append(features_phrase(ph[j], j, n))
            ph_ppl.append(ppl_phrases[j])
            ph_idx.append(decalage + i)
        debut += n

    dim = emb.shape[1] if emb.ndim == 2 else 0
    return {
        "emb_moy": np.asarray(emb_moy, dtype=np.float16).reshape(len(textes), dim),
        "style": calculer_style(textes, n_jobs=n_jobs),
        "coh": np.asarray(coh, dtype=np.float32).reshape(len(textes), N_FEATURES_COHERENCE),
        "ppl": np.asarray(ppl, dtype=np.float32).reshape(len(textes), N_FEATURES_PPL),
        "ph_emb": np.asarray(ph_emb, dtype=np.float16).reshape(len(ph_idx), dim),
        "ph_style": np.asarray(ph_style, dtype=np.float32).reshape(len(ph_idx), N_FEATURES_PHRASE),
        "ph_ppl": np.asarray(ph_ppl, dtype=np.float32).reshape(len(ph_idx), N_FEATURES_PPL_PHRASE),
        "ph_idx": np.asarray(ph_idx, dtype=np.int32),
    }


def _fichiers(dossier: Path, split: str) -> list[Path]:
    return sorted(dossier.glob(f"{split}_[0-9][0-9][0-9][0-9][0-9].npz"))


def charger_split(dossier, split: str) -> dict:
    """
    Relit les features de niveau PARAGRAPHE d'un split.

    Returns:
        dict : emb (embedding moyen des phrases, float16), style, coh, ppl, y (0/1),
               meta (DataFrame)
    """
    dossier = Path(dossier)
    lire_info_cache(dossier)
    chemin_meta = dossier / f"meta_{split}.csv"
    if not chemin_meta.exists():
        raise FileNotFoundError(f"Split '{split}' absent du cache {dossier}")
    meta = pd.read_csv(chemin_meta)

    blocs = [np.load(f) for f in _fichiers(dossier, split)]
    emb = np.concatenate([b["emb_moy"] for b in blocs])
    if len(emb) != len(meta):
        raise ValueError(
            f"Cache incomplet pour '{split}' : {len(emb)} lignes calculées sur {len(meta)}. "
            f"Relancez 'detection preparer' pour terminer le calcul.")
    return {"emb": emb,
            "style": np.concatenate([b["style"] for b in blocs]),
            "coh": np.concatenate([b["coh"] for b in blocs]),
            "ppl": np.concatenate([b["ppl"] for b in blocs]),
            "y": meta["label"].to_numpy(dtype=int), "meta": meta}


def charger_phrases(dossier, split: str, max_exemples: int | None = None, seed: int = 0) -> dict:
    """
    Relit les features de niveau PHRASE d'un split. Chaque phrase hérite du label de
    son paragraphe.

    Args:
        max_exemples : sous-échantillonne au hasard à environ ce nombre de phrases
                       (bloc par bloc, pour limiter la mémoire)
    Returns:
        dict : emb (float16), style, ppl, y, idx_texte (n° du paragraphe de chaque phrase)
    """
    dossier = Path(dossier)
    lire_info_cache(dossier)
    meta = pd.read_csv(dossier / f"meta_{split}.csv")
    fichiers = _fichiers(dossier, split)

    total = sum(len(np.load(f)["ph_idx"]) for f in fichiers)
    frac = 1.0 if not max_exemples or max_exemples >= total else max_exemples / total
    rng = np.random.default_rng(seed)

    emb, style, ppl, idx = [], [], [], []
    for f in fichiers:
        b = np.load(f)
        garde = rng.random(len(b["ph_idx"])) < frac if frac < 1.0 else slice(None)
        emb.append(b["ph_emb"][garde])
        style.append(b["ph_style"][garde])
        ppl.append(b["ph_ppl"][garde])
        idx.append(b["ph_idx"][garde])
    idx = np.concatenate(idx)
    if len(idx) and idx.max() >= len(meta):
        raise ValueError(f"Cache incomplet ou incohérent pour '{split}'.")
    return {"emb": np.concatenate(emb), "style": np.concatenate(style), "ppl": np.concatenate(ppl),
            "y": meta["label"].to_numpy(dtype=int)[idx], "idx_texte": idx}
