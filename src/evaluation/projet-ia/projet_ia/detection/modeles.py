"""
detection/modeles.py
====================
Accès paresseux aux modèles utilisés par la détection IA.
"""

from .. import config
from ..common.models import get_spacy, get_nli


def nlp(texte: str, langue: str = "fr"):
    """Pipeline spaCy de la langue demandée ("fr" : fr_core_news_sm, "en" : en_core_web_sm)."""
    modele = config.LANGUES[config.verifier_langue(langue)]["spacy"]
    if modele is None:
        raise ValueError(f"Pas de modèle spaCy pour la langue '{langue}' (utiliser 'fr' ou 'en').")
    return get_spacy(modele)(texte)


def classifier_nli(*args, **kwargs):
    """Classifieur NLI zero-shot (premier modèle disponible de la liste)."""
    return get_nli(config.MODELES_NLI_DETECTION)(*args, **kwargs)
