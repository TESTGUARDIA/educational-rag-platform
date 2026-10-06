"""
verification/modeles.py
=======================
Accès paresseux aux modèles utilisés par la vérification de réponses.
`nlp(...)` et `nli(...)` s'utilisent comme avant, mais le modèle n'est chargé
qu'au premier appel (voir common/models.py).
"""

from .. import config
from ..common.models import get_spacy, get_nli


def nlp(texte: str):
    """Pipeline spaCy de la vérification (fr_core_news_md)."""
    return get_spacy(config.SPACY_VERIFICATION)(texte)


def nli(*args, **kwargs):
    """Classifieur NLI zero-shot (DeBERTa) de la vérification."""
    return get_nli(config.MODELES_NLI_VERIFICATION)(*args, **kwargs)
