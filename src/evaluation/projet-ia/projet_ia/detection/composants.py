"""
detection/composants.py
=======================
Les trois points de vue (« composants ») qui analysent un paragraphe, chacun avec
son propre MLP :

  paragraphe    embedding moyen des phrases + style global + cohérence + perplexité
  phrase        chaque phrase prise isolément (embedding + style + perplexité de la
                phrase) ; les phrases héritent du label de leur paragraphe
  interphrases  comparaison des phrases entre elles : régularité de longueur,
                de ponctuation, de vocabulaire, de perplexité, cohérence sémantique…
                (pas d'embedding)

Leurs sorties sont ensuite combinées par fusion.py.
"""

from .cache import charger_phrases, charger_split
from .mlp import GenomeMLP
from .style import INDICES_INTERPHRASES

COMPOSANTS = ("paragraphe", "phrase", "interphrases")

# Groupes de features que chaque composant peut utiliser
BLOCS = {
    "paragraphe":   ("emb", "style", "coh", "ppl"),
    "phrase":       ("emb", "style", "ppl"),
    "interphrases": ("style", "coh", "ppl"),
}


def genome_par_defaut(composant: str) -> GenomeMLP:
    """Génome raisonnable, utilisé tant que l'algorithme génétique n'en a pas trouvé un meilleur."""
    if composant == "paragraphe":
        return GenomeMLP(n_couches=2, largeur=256, use_embedding=True, use_style=True,
                         use_coherence=True, use_ppl=True)
    if composant == "phrase":
        return GenomeMLP(n_couches=2, largeur=256, use_embedding=True, use_style=True,
                         use_coherence=False, use_ppl=True)
    if composant == "interphrases":
        return GenomeMLP(n_couches=2, largeur=64, use_embedding=False, use_style=True,
                         use_coherence=True, use_ppl=True)
    raise ValueError(f"Composant inconnu : {composant} (choix : {COMPOSANTS})")


def charger_donnees(dossier_cache, composant: str, split: str,
                    max_exemples: int | None = None, seed: int = 0) -> dict:
    """
    Charge les features d'un composant pour un split.

    Returns:
        dict : emb, style, coh, ppl (None si le composant n'en a pas), y, et
               idx_texte (composant "phrase" : paragraphe d'origine de chaque phrase)
               ou meta (autres composants : DataFrame domaine/longueur/…)
    """
    if composant == "phrase":
        d = charger_phrases(dossier_cache, split, max_exemples=max_exemples, seed=seed)
        return {"emb": d["emb"], "style": d["style"], "coh": None, "ppl": d["ppl"],
                "y": d["y"], "idx_texte": d["idx_texte"]}
    d = charger_split(dossier_cache, split)
    if composant == "interphrases":
        return {"emb": None, "style": d["style"][:, list(INDICES_INTERPHRASES)],
                "coh": d["coh"], "ppl": d["ppl"], "y": d["y"], "meta": d["meta"]}
    if composant == "paragraphe":
        return d
    raise ValueError(f"Composant inconnu : {composant} (choix : {COMPOSANTS})")
