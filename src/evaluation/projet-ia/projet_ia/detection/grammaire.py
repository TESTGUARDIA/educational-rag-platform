"""
detection/grammaire.py
======================
Analyse grammaticale des phrases : catégories POS, statistiques, affichage.
"""

from .modeles import nlp


# Mapping catégories POS spaCy → noms lisibles
POS_CATEGORIES = {
    "NOUN":  "noms_communs",            "PROPN": "noms_propres",
    "VERB":  "verbes",                  "AUX":   "auxiliaires",
    "ADJ":   "adjectifs",               "ADV":   "adverbes",
    "DET":   "determinants",            "PRON":  "pronoms",
    "ADP":   "prepositions",            "CCONJ": "conjonctions_coordination",
    "SCONJ": "conjonctions_subordination", "NUM": "numéraux",
    "INTJ":  "interjections",           "PUNCT": "ponctuation",
    "SPACE": "espaces",                 "SYM":   "symboles",
    "X":     "inconnus",
}


# Catégories qui comptent comme "mots" (hors ponctuation, espaces, symboles)
CATEGORIES_MOTS = [
    "noms_communs", "noms_propres", "verbes", "auxiliaires", "adjectifs",
    "adverbes", "determinants", "pronoms", "prepositions",
    "conjonctions_coordination", "conjonctions_subordination",
    "numéraux", "interjections", "inconnus",
]


def analyser_phrase(phrase: str, langue: str = "fr") -> dict:
    """
    Analyse une phrase et classe chaque token par catégorie grammaticale.

    Returns:
        dict {categorie: [{"texte": ..., "lemme": ...}, ...]} pour les mots,
             {categorie: [texte, ...]} pour la ponctuation/espaces/symboles.
    """
    doc       = nlp(phrase, langue)
    resultats = {cat: [] for cat in POS_CATEGORIES.values()}

    for token in doc:
        categorie = POS_CATEGORIES.get(token.pos_, "inconnus")
        if token.pos_ in ("PUNCT", "SPACE", "SYM"):
            resultats[categorie].append(token.text)
        else:
            resultats[categorie].append({"texte": token.text, "lemme": token.lemma_})

    return resultats


def calculer_statistiques(analyse: dict) -> dict:
    """
    Calcule les statistiques d'une phrase à partir du dictionnaire d'analyse.

    Returns:
        dict avec total_mots, total_ponctuation,
              compte_par_categorie, pourcentage_par_categorie.
    """
    compte = {cat: len(analyse[cat]) for cat in CATEGORIES_MOTS}
    total  = sum(compte.values())

    pourcentages = {
        cat: round(c / total * 100, 1) if total > 0 else 0.0
        for cat, c in compte.items()
    }

    return {
        "total_mots":               total,
        "total_ponctuation":        len(analyse.get("ponctuation", [])),
        "compte_par_categorie":     compte,
        "pourcentage_par_categorie": pourcentages,
    }


def afficher_statistiques(phrase: str, langue: str = "fr") -> None:
    """Affiche les statistiques grammaticales d'une phrase dans le terminal."""
    analyse = analyser_phrase(phrase, langue)
    stats   = calculer_statistiques(analyse)

    labels = {
        "noms_communs":               "Noms communs             ",
        "noms_propres":               "Noms propres             ",
        "verbes":                     "Verbes                   ",
        "auxiliaires":                "Auxiliaires              ",
        "adjectifs":                  "Adjectifs                ",
        "adverbes":                   "Adverbes                 ",
        "determinants":               "Déterminants             ",
        "pronoms":                    "Pronoms                  ",
        "prepositions":               "Prépositions             ",
        "conjonctions_coordination":  "Conjonctions coord.      ",
        "conjonctions_subordination": "Conjonctions subord.     ",
        "numéraux":                   "Numéraux                 ",
        "interjections":              "Interjections            ",
        "inconnus":                   "Inconnus                 ",
    }

    print(f"\n{'='*55}")
    print(f"Phrase : {phrase}")
    print(f"{'='*55}")
    print(f"  Total mots        : {stats['total_mots']}")
    print(f"  Total ponctuation : {stats['total_ponctuation']}")
    print(f"\n  Détail par catégorie (mots non nuls) :")
    print(f"  {'-'*50}")
    for cat, label in labels.items():
        compte = stats["compte_par_categorie"][cat]
        if compte == 0:
            continue
        pct   = stats["pourcentage_par_categorie"][cat]
        barre = "█" * int(pct / 5)
        print(f"  {label} : {compte:>3}  ({pct:>5}%)  {barre}")
    print(f"  {'-'*50}")


def stats_phrase(phrase: str, langue: str = "fr") -> dict:
    """
    Statistiques simples d'une phrase, utilisées comme features par l'algorithme
    génétique : nombre de mots et nombre de signes de ponctuation.
    """
    doc = nlp(str(phrase), langue)
    return {
        "total_mots":  sum(1 for t in doc if not t.is_punct and not t.is_space),
        "total_ponct": sum(1 for t in doc if t.is_punct),
    }


def stats_corpus(phrases, langue: str = "fr") -> list[dict]:
    """Applique stats_phrase() à toutes les phrases (liste, Series pandas…)."""
    return [stats_phrase(p, langue) for p in phrases]
