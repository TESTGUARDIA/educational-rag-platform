"""
detection/relations.py
======================
Analyse sémantique des relations syntaxiques entre les mots (NLI zero-shot).
"""

import pandas as pd

from .modeles import nlp, classifier_nli


# Labels sémantiques par catégorie POS.
# Le pré-filtrage réduit le nombre d'inférences : au lieu de tester tous les
# labels sur chaque token, on ne teste que ceux pertinents pour son POS.
LABELS_PAR_POS = {
    "VERB": [
        "action physique", "action mentale", "action de communication",
        "action de mouvement", "action de création", "action de destruction",
        "action de transfert", "état", "changement d'état",
        "relation causale", "relation affective", "cause", "conséquence", "but",
    ],
    "AUX":  ["état", "action physique", "changement d'état"],
    "NOUN": [
        "possession", "appartenance à un groupe", "parenté", "relation sociale",
        "relation professionnelle", "relation causale", "domaine scientifique",
        "domaine juridique", "domaine économique", "domaine médical",
        "domaine technologique", "domaine culturel",
        "localisation géographique", "localisation abstraite", "quantité",
    ],
    "PROPN": [
        "possession", "parenté", "relation sociale", "relation professionnelle",
        "appartenance à un groupe", "localisation géographique", "origine",
    ],
    "ADJ": [
        "description physique", "description temporelle",
        "description émotionnelle", "description abstraite",
        "intensité", "relation de comparaison", "relation d'opposition",
        "domaine scientifique", "domaine médical",
    ],
    "ADV": [
        "intensité", "fréquence", "moment précis", "durée", "séquence",
        "cause", "conséquence", "condition", "concession",
    ],
    "ADP": [
        "localisation géographique", "localisation relative", "direction",
        "origine", "moment précis", "durée", "cause", "but",
        "condition", "possession",
    ],
    "CCONJ": [
        "relation d'opposition", "relation de comparaison",
        "cause", "conséquence", "concession", "séquence",
    ],
    "SCONJ": [
        "cause", "conséquence", "condition", "concession",
        "but", "hypothèse", "relation temporelle",
    ],
    "DET":  ["possession", "quantité", "description abstraite"],
    "PRON": ["relation sociale", "possession", "appartenance à un groupe"],
    "NUM":  ["quantité", "moment précis", "durée"],
    "INTJ": ["description émotionnelle", "relation affective", "état"],
}


LABELS_DEFAUT = ["description abstraite", "relation causale", "état"]


# Descriptions lisibles des dépendances syntaxiques spaCy
DESCRIPTIONS_DEP = {
    "nsubj": "sujet de", "obj": "objet direct de", "iobj": "objet indirect de",
    "amod": "adjectif qualifiant", "advmod": "adverbe modifiant",
    "det": "déterminant de", "prep": "préposition liée à",
    "pobj": "objet de la préposition", "aux": "auxiliaire de",
    "aux:pass": "auxiliaire passif de", "cop": "copule de",
    "nmod": "modificateur nominal de", "appos": "apposition de",
    "conj": "coordonné à", "cc": "conjonction de",
    "mark": "subordonnant de", "acl": "proposition relative de",
    "xcomp": "complément verbal de", "ccomp": "complétive de",
    "ROOT": "racine de la phrase",
}


def extraire_relations(phrase: str) -> pd.DataFrame:
    """
    Extrait les relations syntaxiques et sémantiques entre les mots d'une phrase.

    Pour chaque token (hors ponctuation/espaces) :
      - Sa catégorie POS et son token "tête" dans l'arbre syntaxique
      - La description de la relation de dépendance
      - Le sens sémantique prédit par NLI (label le plus probable)

    Returns:
        DataFrame avec colonnes : mot, pos, tete, pos_tete, relation_dep,
                                   description, sens, score
    """
    doc     = nlp(phrase)
    lignes  = []
    contextes = []
    labels_batch = []

    for token in doc:
        if token.pos_ in ("PUNCT", "SPACE"):
            continue
        dep         = token.dep_
        description = DESCRIPTIONS_DEP.get(dep, f"lié à ({dep})")
        contexte    = f"{token.text} {description} {token.head.text}"
        labels      = LABELS_PAR_POS.get(token.pos_, LABELS_DEFAUT)

        contextes.append(contexte)
        labels_batch.append(labels)
        lignes.append({
            "mot":          token.text,
            "pos":          token.pos_,
            "tete":         token.head.text,
            "pos_tete":     token.head.pos_,
            "relation_dep": dep,
            "description":  description,
        })

    sens_lst  = []
    score_lst = []
    for contexte, labels in zip(contextes, labels_batch):
        res = classifier_nli(contexte, labels)
        sens_lst.append(res["labels"][0])
        score_lst.append(round(res["scores"][0], 3))

    for i, ligne in enumerate(lignes):
        ligne["sens"]  = sens_lst[i]
        ligne["score"] = score_lst[i]

    return pd.DataFrame(lignes)
