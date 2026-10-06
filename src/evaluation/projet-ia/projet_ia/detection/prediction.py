"""
detection/prediction.py
=======================
Prédiction « IA ou humain » pour un paragraphe (français ou anglais).

Le texte est découpé en phrases, chaque phrase est encodée, puis trois analyses sont
menées (paragraphe entier, phrase par phrase, comparaison entre phrases) et fusionnées
avec les poids appris à l'entraînement. Le résultat détaille aussi chaque phrase.

La langue est détectée automatiquement (langue="auto") pour choisir le bon modèle.
"""

import pickle
from pathlib import Path

import numpy as np

from .. import config
from ..calibration.regression import sigmoid
from .cache import decouper_pour_cache, encodeur_depuis_info
from .fusion import agreger_phrases, assembler_entrees, logit, predire_fusion
from .langue import detecter_langue
from .mlp import GenomeMLP, assembler_features, predire_proba
from .style import (INDICES_INTERPHRASES, calculer_style, coherence_depuis_embeddings,
                    features_phrase)

MIN_PHRASES_FIABLE = 3


def charger_modele(chemin) -> dict:
    """Charge un modèle sauvegardé par entrainer_final()."""
    with open(chemin, "rb") as f:
        return pickle.load(f)


def _choisir_modele(langue: str) -> tuple[Path, str | None]:
    """Fichier modèle pour une langue, avec repli sur le modèle multilingue. Retourne (chemin, note)."""
    direct = config.CheminsDetection(langue).modele
    if direct.exists():
        return direct, None
    multi = config.CheminsDetection("multi").modele
    if multi.exists():
        return multi, f"Pas de modèle '{langue}' : modèle multilingue utilisé."
    raise FileNotFoundError(
        f"Aucun modèle pour la langue '{langue}' ({direct}). Entraînez-le avec : "
        f"python -m projet_ia detection genetique --langue {langue}")


def _predire_avec_modele(textes: list[str], data: dict, encodeur, seuil: float) -> list[dict]:
    """Applique un modèle chargé à une liste de textes."""
    info = data["cache"]
    genomes = {c: GenomeMLP.from_dict(d["genome"]) for c, d in data["composants"].items()}
    encodeur = encodeur or encodeur_depuis_info(info)

    phrases = [decouper_pour_cache(t, info["max_phrases"]) for t in textes]
    emb = np.asarray(encodeur([p for ph in phrases for p in ph]), dtype=np.float32)

    emb_moy, coh, ph_style, idx = [], [], [], []
    debut = 0
    for i, ph in enumerate(phrases):
        e = emb[debut:debut + len(ph)]
        emb_moy.append(e.mean(axis=0))
        coh.append(coherence_depuis_embeddings(e))
        ph_style.extend(features_phrase(p, j, len(ph)) for j, p in enumerate(ph))
        idx.extend([i] * len(ph))
        debut += len(ph)
    emb_moy, coh, ph_style = np.array(emb_moy), np.array(coh), np.array(ph_style)
    style = calculer_style(textes)

    comp = data["composants"]
    z_phrases = logit(predire_proba(
        comp["phrase"], assembler_features(emb, ph_style, None, genomes["phrase"])))
    z = {
        "paragraphe": logit(predire_proba(comp["paragraphe"], assembler_features(
            emb_moy, style, coh, genomes["paragraphe"]))),
        "interphrases": logit(predire_proba(comp["interphrases"], assembler_features(
            None, style[:, list(INDICES_INTERPHRASES)], coh, genomes["interphrases"]))),
        **agreger_phrases(z_phrases, idx, len(textes)),
    }
    proba = predire_fusion(data["fusion"], assembler_entrees(z))

    resultats, debut = [], 0
    for i, (texte, p) in enumerate(zip(textes, proba)):
        n = len(phrases[i])
        label = int(p >= seuil)
        r = {
            "prediction": "IA" if label else "humain", "label": label,
            "probabilite_ia": round(float(p), 4),
            "confiance": round(float(p if label else 1 - p), 4),
            "langue": info["langue"], "n_phrases": n,
            "composants": {c: round(float(sigmoid(z[c][i])), 4)
                           for c in ("paragraphe", "interphrases", "phrases_moy")},
            "phrases": [{"texte": ph, "probabilite_ia": round(float(sigmoid(zp)), 4)}
                        for ph, zp in zip(phrases[i], z_phrases[debut:debut + n])],
        }
        if n < MIN_PHRASES_FIABLE:
            r["avertissement"] = (f"Texte court ({n} phrase(s)) : la comparaison entre phrases "
                                  f"est peu informative, fiabilité réduite.")
        resultats.append(r)
        debut += n
    return resultats


def predire_textes(textes, langue: str = "auto", chemin_modele=None, encodeur=None,
                   seuil: float = 0.5) -> list[dict]:
    """
    Prédit pour une liste de paragraphes.

    Args:
        langue        : "auto" (détection pour chaque texte), "en", "fr" ou "multi"
        chemin_modele : forcer un fichier modèle précis (désactive le choix par langue)
        encodeur      : fonction textes -> embeddings (par défaut : celui du modèle)
    Returns:
        liste de dict (dans l'ordre des textes) : prediction ("IA"/"humain"), label,
        probabilite_ia, confiance, langue, n_phrases, composants (probabilité de
        chaque analyse), phrases (probabilité de chaque phrase), et éventuellement
        `avertissement`.
    """
    textes = list(textes)
    if chemin_modele is not None:
        return _predire_avec_modele(textes, charger_modele(chemin_modele), encodeur, seuil)

    if langue == "auto":
        langues = [detecter_langue(t) for t in textes]
    else:
        langues = [config.verifier_langue(langue)] * len(textes)

    resultats = [None] * len(textes)
    for l in set(langues):
        voulue = l or config.LANGUE_PAR_DEFAUT
        chemin, note = _choisir_modele(voulue)
        if l is None:
            note = (note + " " if note else "") + \
                   f"Langue non détectée : modèle '{voulue}' utilisé par défaut."
        pos = [i for i, x in enumerate(langues) if x == l]
        sorties = _predire_avec_modele([textes[i] for i in pos], charger_modele(chemin),
                                       encodeur, seuil)
        for i, r in zip(pos, sorties):
            if note:
                r["avertissement"] = (r.get("avertissement", "") + " " + note).strip()
            resultats[i] = r
    return resultats


def predire_depuis_fichier(texte: str, langue: str = "auto", chemin_modele=None,
                           encodeur=None) -> dict:
    """Prédit pour un seul texte (voir predire_textes)."""
    return predire_textes([texte], langue, chemin_modele, encodeur)[0]
