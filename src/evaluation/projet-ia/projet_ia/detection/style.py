"""
detection/style.py
==================
Features de STYLE calculées sur un paragraphe, dont la comparaison des phrases
entre elles (régularité de la longueur, de la ponctuation, du vocabulaire…).
Les textes écrits par une IA sont souvent plus réguliers d'une phrase à l'autre
que ceux d'un humain : ces features visent à capter cette différence.

Deux familles :
  * features_style()       : rapides, sans modèle (regex), 20 valeurs par texte
  * calculer_coherence()   : optionnelles, basées sur les embeddings de chaque
                             phrase (similarité entre phrases successives).
                             Coûteuses : chaque phrase passe dans l'encodeur.

Ne dépend d'aucune langue en particulier (pas de spaCy, pas de dictionnaire).
"""

import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_FIN_PHRASE = re.compile(r"(?<=[.!?…])\s+")
_MOT = re.compile(r"\w+", re.UNICODE)
_PONCT = re.compile(r"[^\w\s]", re.UNICODE)

NOMS_FEATURES_STYLE = [
    # ── globales ──
    "n_mots", "n_phrases", "longueur_moy_mot", "ratio_mots_uniques",
    "ratio_ponctuation", "ratio_virgules", "ratio_majuscules",
    # ── comparaison des phrases entre elles ──
    "phrase_lg_moy", "phrase_lg_std", "phrase_lg_cv", "phrase_lg_min", "phrase_lg_max",
    "phrase_burstiness", "ponct_std_entre_phrases", "longueur_mot_std_entre_phrases",
    "jaccard_successives_moy", "jaccard_successives_std",
    "ratio_debuts_repetes", "ratio_phrases_courtes", "ratio_phrases_longues",
]
N_FEATURES_STYLE = len(NOMS_FEATURES_STYLE)

# Sous-ensemble de NOMS_FEATURES_STYLE qui compare les phrases entre elles
# (n_phrases + toutes les features "entre phrases") : entrée du modèle « interphrases ».
INDICES_INTERPHRASES = (1,) + tuple(range(7, N_FEATURES_STYLE))

# Features d'UNE phrase prise isolément : entrée du modèle « phrase ».
NOMS_FEATURES_PHRASE = ["n_mots", "longueur_moy_mot", "ratio_ponctuation", "virgules_par_mot",
                        "ratio_majuscules", "ratio_chiffres", "position_relative"]
N_FEATURES_PHRASE = len(NOMS_FEATURES_PHRASE)

NOMS_FEATURES_COHERENCE = ["coh_sim_moy", "coh_sim_std", "coh_sim_min", "coh_dist_centroide_moy"]
N_FEATURES_COHERENCE = len(NOMS_FEATURES_COHERENCE)


def decouper_phrases(texte: str) -> list[str]:
    """Découpe un texte en phrases (sur . ! ? …). Retombe sur les retours à la ligne
    pour la poésie ou les textes sans ponctuation."""
    phrases = [p.strip() for p in _FIN_PHRASE.split(texte) if p.strip()]
    if len(phrases) <= 1 and "\n" in texte:
        phrases = [l.strip() for l in texte.split("\n") if l.strip()]
    return phrases


def features_style(texte: str) -> np.ndarray:
    """Vecteur de N_FEATURES_STYLE features de style pour un texte (voir NOMS_FEATURES_STYLE)."""
    phrases = decouper_phrases(texte)
    mots_par_phrase = [_MOT.findall(p.lower()) for p in phrases]
    mots_par_phrase = [m for m in mots_par_phrase if m]          # ignore les phrases sans mot
    if not mots_par_phrase:
        return np.zeros(N_FEATURES_STYLE, dtype=np.float32)

    n_ph = len(mots_par_phrase)
    lg = np.array([len(m) for m in mots_par_phrase], dtype=float)
    tous = [w for m in mots_par_phrase for w in m]
    n_mots = len(tous)

    lg_mot_par_phrase = np.array([np.mean([len(w) for w in m]) for m in mots_par_phrase])
    ponct_par_phrase = np.array([len(_PONCT.findall(p)) for p in phrases],
                                dtype=float)
    n_car = max(len(texte), 1)
    n_lettres = max(sum(c.isalpha() for c in texte), 1)

    lg_moy, lg_std = lg.mean(), lg.std()
    burst = (lg_std - lg_moy) / (lg_std + lg_moy) if (lg_std + lg_moy) > 0 else 0.0

    ensembles = [set(m) for m in mots_par_phrase]
    jaccards = [len(a & b) / len(a | b) for a, b in zip(ensembles, ensembles[1:])]
    debuts = [m[0] for m in mots_par_phrase]

    return np.array([
        n_mots,
        n_ph,
        np.mean([len(w) for w in tous]),
        len(set(tous)) / n_mots,
        len(_PONCT.findall(texte)) / n_car,
        texte.count(",") / n_mots,
        sum(c.isupper() for c in texte) / n_lettres,
        lg_moy,
        lg_std,
        lg_std / lg_moy if lg_moy > 0 else 0.0,
        lg.min(),
        lg.max(),
        burst,
        ponct_par_phrase.std(),
        lg_mot_par_phrase.std(),
        float(np.mean(jaccards)) if jaccards else 0.0,
        float(np.std(jaccards)) if jaccards else 0.0,
        1.0 - len(set(debuts)) / n_ph,
        float(np.mean(lg < 5)),
        float(np.mean(lg > 25)),
    ], dtype=np.float32)


def _style_lot(textes: list[str]) -> np.ndarray:
    return np.stack([features_style(t) for t in textes]) if textes else \
        np.zeros((0, N_FEATURES_STYLE), dtype=np.float32)


def calculer_style(textes, n_jobs: int = 1, taille_lot: int = 2000) -> np.ndarray:
    """
    Features de style pour une liste de textes → tableau (n_textes, N_FEATURES_STYLE).
    n_jobs > 1 répartit le calcul sur plusieurs processus (utile sur de gros corpus).
    """
    textes = list(textes)
    if n_jobs <= 1 or len(textes) <= taille_lot:
        return _style_lot(textes)
    lots = [textes[i:i + taille_lot] for i in range(0, len(textes), taille_lot)]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        return np.concatenate(list(pool.map(_style_lot, lots)))


def features_phrase(phrase: str, position: int = 0, n_phrases: int = 1) -> np.ndarray:
    """Features d'une phrase isolée (voir NOMS_FEATURES_PHRASE)."""
    mots = _MOT.findall(phrase)
    n_car = max(len(phrase), 1)
    n_mots = max(len(mots), 1)
    return np.array([
        len(mots),
        np.mean([len(m) for m in mots]) if mots else 0.0,
        len(_PONCT.findall(phrase)) / n_car,
        phrase.count(",") / n_mots,
        sum(c.isupper() for c in phrase) / max(sum(c.isalpha() for c in phrase), 1),
        sum(c.isdigit() for c in phrase) / n_car,
        position / max(n_phrases - 1, 1),
    ], dtype=np.float32)


def coherence_depuis_embeddings(emb: np.ndarray) -> np.ndarray:
    """
    Cohérence sémantique d'UN paragraphe à partir des embeddings de ses phrases
    (tableau (n_phrases, dim)) :
        coh_sim_moy / coh_sim_std / coh_sim_min : similarité cosinus entre phrases successives
        coh_dist_centroide_moy : distance moyenne des phrases au centre du paragraphe
    """
    emb = np.asarray(emb, dtype=np.float32)
    if len(emb) < 2:                         # une seule phrase : trivialement cohérent
        return np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)
    sims = np.sum(e[:-1] * e[1:], axis=1)
    centre = e.mean(axis=0)
    centre = centre / max(np.linalg.norm(centre), 1e-9)
    return np.array([sims.mean(), sims.std(), sims.min(), np.mean(1.0 - e @ centre)],
                    dtype=np.float32)


def calculer_coherence(textes, encodeur, max_phrases: int = 30) -> np.ndarray:
    """
    Cohérence entre phrases pour une liste de textes (encode chaque phrase).

    Args:
        encodeur    : fonction liste_de_textes -> tableau (n, dim) (ex. encoder_textes)
        max_phrases : nombre maximum de phrases analysées par texte
    Returns:
        tableau (n_textes, N_FEATURES_COHERENCE)
    """
    textes = list(textes)
    liste_phrases, bornes = [], []
    for t in textes:
        ph = decouper_phrases(t)[:max_phrases] or [t]
        bornes.append((len(liste_phrases), len(liste_phrases) + len(ph)))
        liste_phrases.extend(ph)
    emb = np.asarray(encodeur(liste_phrases), dtype=np.float32) if liste_phrases else \
        np.zeros((0, 1), dtype=np.float32)
    return np.stack([coherence_depuis_embeddings(emb[a:b]) for a, b in bornes]) if bornes else \
        np.zeros((0, N_FEATURES_COHERENCE), dtype=np.float32)
