"""
verification/utilitaires.py
===========================
Utilitaires linguistiques communs : poids grammaticaux, détection de négation,
extraction des mots importants.
"""

from .modeles import nlp


# Poids grammaticaux pour la pondération des embeddings.
# Les mots à fort contenu sémantique reçoivent un poids élevé ;
# les mots grammaticaux (déterminants, pronoms, conjonctions) sont ignorés.
POIDS_POS = {
    "PROPN": 10.0,  # noms propres   : poids dominant (Paris, Shakespeare…)
    "NOUN":   2.5,  # noms communs   : portent l'essentiel du sens
    "NUM":    2.5,  # nombres        : critiques dans les réponses factuelles
    "VERB":   1.0,  # verbes         : utiles mais secondaires
    "ADJ":    0.5,  # adjectifs      : précisent le sens
    "ADV":    0.2,  # adverbes       : peu importants
    "DET":    0.0, "PRON":  0.0, "ADP":   0.0, "AUX":   0.0,
    "CCONJ":  0.0, "SCONJ": 0.0, "PUNCT": 0.0, "SPACE": 0.0, "X": 0.0,
}


def detecter_negation(texte: str) -> bool:
    """Retourne True si le texte contient une négation grammaticale (dep_ == 'neg')."""
    doc = nlp(texte)
    return any(token.dep_ == "neg" for token in doc)


def mots_importants(texte: str) -> list[dict]:
    """
    Retourne les mots importants d'un texte avec leur poids grammatical.
    Utile pour le débogage et l'explication du score.
    """
    doc = nlp(texte)
    return [
        {"mot": t.text, "pos": t.pos_, "poids": POIDS_POS.get(t.pos_, 0.0)}
        for t in doc if POIDS_POS.get(t.pos_, 0.0) > 0
    ]
