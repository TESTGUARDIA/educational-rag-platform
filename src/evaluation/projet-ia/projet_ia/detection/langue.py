"""
detection/langue.py
===================
Détection légère de la langue (français / anglais) par comptage de mots-outils.
Sans dépendance : suffisante pour router un texte vers le bon modèle.
"""

import re

_MOTS = {
    "en": {"the", "and", "of", "to", "in", "is", "that", "it", "was", "for", "with", "as", "on",
           "are", "this", "be", "by", "at", "not", "have", "from", "but", "they", "his", "her",
           "you", "which", "were", "has", "had", "their", "will", "would", "there", "been"},
    "fr": {"le", "la", "les", "de", "des", "du", "et", "est", "en", "un", "une", "que", "qui",
           "dans", "pour", "pas", "sur", "au", "aux", "ce", "se", "il", "elle", "nous", "vous",
           "ils", "sont", "avec", "plus", "par", "mais", "ou", "été", "être", "leur", "cette"},
}
_TOKEN = re.compile(r"[a-zà-öø-ÿœ']+", re.IGNORECASE)


def detecter_langue(texte: str) -> str | None:
    """
    Retourne "en" ou "fr", ou None si le texte est trop court / ambigu pour trancher.
    """
    mots = [m.lower().strip("'") for m in _TOKEN.findall(texte)]
    if len(mots) < 5:
        return None
    score = {l: sum(m in ens for m in mots) for l, ens in _MOTS.items()}
    # Les accents typiquement français renforcent le score français
    score["fr"] += 0.5 * sum(texte.count(c) for c in "éèêàçùôî")
    meilleur = max(score, key=score.get)
    autre = min(score, key=score.get)
    if score[meilleur] < 2 or score[meilleur] < 1.5 * score[autre]:
        return None
    return meilleur
