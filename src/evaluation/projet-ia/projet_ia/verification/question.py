"""
verification/question.py
========================
Analyse linguistique de la question : type, caractère (fermée / semi-ouverte /
ouverte), concept clé attendu, résolution des coréférences, découpage en
sous-questions.
"""

import re

from .modeles import nlp


def detecter_caractere_quel(question: str) -> str:
    """
    Affine le caractère d'une question "Quel/Quelle" :
      - "fermée"      : attend une valeur précise (ex: "Quelle est la capitale de…")
      - "semi-ouverte": attend un développement  (ex: "Quel est le rôle de…")
    """
    TERMES_DEVELOPPEMENT = {
        "rôle", "importance", "fonction", "utilité", "effet", "impact",
        "conséquence", "influence", "principe", "signification", "sens",
        "définition", "intérêt", "avantage", "inconvénient", "différence",
        "rapport", "lien", "relation", "caractéristique", "particularité",
        "mécanisme", "processus", "fonctionnement", "but", "objectif",
        "cause", "raison", "explication", "description", "nature",
    }
    doc = nlp(question.lower())
    for i, token in enumerate(doc[:-1]):
        if token.pos_ == "DET" and doc[i + 1].pos_ == "NOUN":
            if doc[i + 1].lemma_ in TERMES_DEVELOPPEMENT:
                return "semi-ouverte"
    return "fermée"


def analyser_question(question: str) -> dict:
    """
    Analyse la question pour en extraire :
      - type_question      : "personne", "lieu", "date", "quantité", "concept", "autre"
      - caractere_question : "fermée", "semi-ouverte", "ouverte"
      - pos_prioritaires   : catégories POS à sur-pondérer dans la réponse
      - multiplicateur     : facteur appliqué sur ces catégories
      - concept_cle        : concept sémantique attendu dans la réponse (ou None)
      - mots_cles_question : mots importants de la question (contexte)
    """
    doc = nlp(question.lower())

    # Mapping mot interrogatif → (pos_prioritaires, multiplicateur, caractere, type)
    INTERROGATIFS = {
        "qui":      (["PROPN"],                8.0, "fermée",       "personne"),
        "lequel":   (["PROPN", "NOUN"],        6.0, "fermée",       "entité"),
        "laquelle": (["PROPN", "NOUN"],        6.0, "fermée",       "entité"),
        "où":       (["PROPN", "NOUN"],        6.0, "fermée",       "lieu"),
        "quand":    (["NUM", "NOUN"],          6.0, "fermée",       "date"),
        "combien":  (["NUM"],                  7.0, "fermée",       "quantité"),
        "quel":     (["NOUN", "PROPN", "ADJ"], 4.0, "semi-ouverte", "concept"),
        "quelle":   (["NOUN", "PROPN", "ADJ"], 4.0, "semi-ouverte", "concept"),
        "quels":    (["NOUN", "PROPN", "ADJ"], 4.0, "semi-ouverte", "concept"),
        "quelles":  (["NOUN", "PROPN", "ADJ"], 4.0, "semi-ouverte", "concept"),
        "comment":  (["VERB", "ADV"],          2.0, "ouverte",      "explication"),
        "pourquoi": (["VERB", "ADJ", "NOUN"],  2.0, "ouverte",      "explication"),
        "quoi":     (["NOUN", "PROPN"],        2.0, "ouverte",      "concept"),
        "que":      (["NOUN", "PROPN"],        2.0, "ouverte",      "concept"),
        "qu":       (["NOUN", "PROPN"],        2.0, "ouverte",      "concept"),
    }

    pos_prioritaires   = ["NOUN", "PROPN"]
    multiplicateur     = 2.0
    type_question      = "autre"
    caractere_question = "ouverte"

    for token in doc:
        forme = token.text.lower().rstrip("'").rstrip("'")
        if forme in INTERROGATIFS:
            pos_prioritaires, multiplicateur, caractere_question, type_question = \
                INTERROGATIFS[forme]
            if forme in ("quel", "quelle", "quels", "quelles"):
                caractere_question = detecter_caractere_quel(question)
            break

    # Détection du concept clé attendu dans la réponse
    concept_cle = detecter_concept_cle(question)
    if concept_cle and type_question in ("concept", "autre"):
        type_question    = "concept"
        pos_prioritaires = ["NOUN"]
        multiplicateur   = max(multiplicateur, 3.0)

    # Mots-clés contextuels de la question
    doc_orig = nlp(question)
    mots_cles_question = [
        {"mot": t.text, "pos": t.pos_, "entite": t.ent_type_ or None}
        for t in doc_orig
        if t.pos_ in ["NOUN", "PROPN", "NUM"]
        and t.text.lower() not in INTERROGATIFS
        and not t.is_stop
    ]

    return {
        "type_question":      type_question,
        "caractere_question": caractere_question,
        "pos_prioritaires":   pos_prioritaires,
        "multiplicateur":     multiplicateur,
        "concept_cle":        concept_cle,
        "mots_cles_question": mots_cles_question,
    }


def detecter_concept_cle(question: str) -> dict | None:
    """
    Détecte le concept sémantique clé attendu dans la réponse en analysant
    la structure de la question (ex: "fruit" pour "Quel est ton fruit préféré ?").
    Retourne None si aucun concept pertinent n'est détecté.
    """
    doc = nlp(question)

    MOTS_EXCLUS = {
        "chose", "fait", "type", "sorte", "genre", "élément",
        "résultat", "réponse", "valeur", "nom", "prénom",
        "rôle", "importance", "fonction", "utilité", "effet", "impact",
        "conséquence", "influence", "principe", "signification", "sens",
        "définition", "intérêt", "avantage", "inconvénient", "différence",
        "rapport", "lien", "relation", "caractéristique", "particularité",
        "mécanisme", "processus", "fonctionnement", "but", "objectif",
        "cause", "raison", "explication", "description", "nature",
    }

    # Pattern 1 : "de/d' + DET? + NOUN" → sujet de la question
    for i, token in enumerate(doc[:-1]):
        if token.lemma_ in ("de", "d") or token.pos_ == "ADP":
            for j in range(i + 1, min(i + 4, len(doc))):
                t = doc[j]
                if (t.pos_ in ("NOUN", "PROPN")
                        and t.lemma_.lower() not in MOTS_EXCLUS
                        and not t.is_stop):
                    return {"mot": t.text, "lemme": t.lemma_, "pos": t.pos_}

    # Pattern 2 : "DET + NOUN"
    for i, token in enumerate(doc[:-1]):
        suivant = doc[i + 1]
        if (token.pos_ == "DET"
                and suivant.pos_ in ("NOUN", "PROPN")
                and suivant.lemma_.lower() not in MOTS_EXCLUS
                and not suivant.is_stop):
            return {"mot": suivant.text, "lemme": suivant.lemma_, "pos": suivant.pos_}

    return None


def resoudre_coreferences(reponse: str, question: str) -> str:
    """
    Résout les coréférences dans la réponse en remplaçant pronoms/démonstratifs
    par le sujet principal extrait de la question.
    Ex: "C'est Paris." → "La capitale est Paris." (si question sur la capitale)
    """
    doc_q   = nlp(question)
    sujet   = next(
        (t.text for t in doc_q if t.pos_ in ("NOUN", "PROPN") and not t.is_stop),
        None
    )
    if not sujet:
        return reponse

    # Normalise les apostrophes
    r = reponse
    for apos in ["'", "'", "`"]:
        r = r.replace(apos, "'")

    patterns = [
        (r"(?i)^[Cc]'est\b",            f"{sujet} est"),
        (r"(?i)^[Cc]e sont\b",          f"{sujet} sont"),
        (r"(?i)^[Ii]l s'agit d[e']\s*", f"{sujet} est "),
        (r"(?i)^[Ii]l\b",               sujet),
        (r"(?i)^[Ee]lle\b",             sujet),
    ]
    for pattern, remplacement in patterns:
        r_new = re.sub(pattern, remplacement, r)
        if r_new != r:
            return r_new
    return r


def detecter_sous_questions(question: str) -> list[str]:
    """
    Détecte si une question contient plusieurs sous-questions liées par "et",
    "ainsi que", etc. Retourne une liste de sous-questions (ou [question] si simple).
    """
    SEPARATEURS = [" et ", " ainsi que ", " mais aussi ", " de même que ", " et aussi "]
    for sep in SEPARATEURS:
        if sep in question.lower():
            parties = re.split(re.escape(sep), question, flags=re.IGNORECASE, maxsplit=1)
            if len(parties) == 2:
                p1 = parties[0].strip().rstrip("?").strip()
                p2 = parties[1].strip().rstrip("?").strip()
                if len(p1.split()) >= 3 and len(p2.split()) >= 2:
                    return [p1, p2]
    return [question]


def decouper_reponse_correction(reponse: str, correction: str, n: int) -> list[tuple]:
    """
    Découpe réponse et correction en n parties en cherchant des séparateurs naturels.
    Retourne une liste de tuples (reponse_i, correction_i).
    """
    def decouper(texte, nb):
        for sep in [r"\s+et\s+", r",\s*", r";\s*", r"\.\s+"]:
            parties = re.split(sep, texte, maxsplit=nb - 1, flags=re.IGNORECASE)
            if len(parties) >= nb:
                return [p.strip() for p in parties[:nb]]
        mots = texte.split()
        mid  = len(mots) // 2
        return [" ".join(mots[:mid]), " ".join(mots[mid:])]

    parties_rep = decouper(reponse, n)
    parties_cor = decouper(correction, n)
    while len(parties_rep) < n:
        parties_rep.append(parties_rep[-1] if parties_rep else reponse)
    while len(parties_cor) < n:
        parties_cor.append(parties_cor[-1] if parties_cor else correction)
    return list(zip(parties_rep, parties_cor))
