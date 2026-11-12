"""
Analyse d'un cours (Markdown + LaTeX) en fiches de notions, partie par partie.

Deux signaux d'importance sont combinés :
  - la structure du cours : blocs « Définition », « Théorème », « Propriété »,
    « Méthode »… repérés par expressions régulières (fiable, gratuit) et donnés au
    modèle comme liste de contrôle pour qu'il n'en oublie aucun ;
  - le jugement du modèle : importance 3 (central, à évaluer), 2 (utile),
    1 (anecdotique : historique, remarques de culture générale).
"""

import re

import httpx

from .ollama import ErreurModele, discuter

TYPES = ["definition", "propriete", "theoreme", "methode", "exemple", "demonstration", "remarque"]

SCHEMA_NOTIONS = {
    "type": "object",
    "properties": {"notions": {"type": "array", "maxItems": 15, "items": {
        "type": "object",
        "properties": {
            "titre": {"type": "string"},
            "type": {"type": "string", "enum": TYPES},
            "enonce": {"type": "string"},
            "importance": {"type": "integer", "minimum": 1, "maximum": 3},
            "mots_cles": {"type": "array", "maxItems": 5, "items": {"type": "string"}},
        },
        "required": ["titre", "type", "enonce", "importance", "mots_cles"]}}},
    "required": ["notions"],
}

_PARTIE = re.compile(r"(?mi)^\s*#{0,3}\s*\**\s*Partie\s+(\d+)\s*\**\s*[:\-–]?\s*(.*?)\**\s*$")
_BLOC = re.compile(r"(?mi)^\s*\**\s*(Définition|Propriétés?|Théorème|Méthode|Démonstration)s?\s*\**\s*:?\s*\**\s*(.{0,90})")

# Pas d'exemple de notion propre à une matière : un petit modèle le recopie (une notion
# « Intégration par parties » est apparue dans un cours de SVT).
CONSIGNE = """Tu analyses une partie d'un cours (matière : {matiere}) pour préparer des évaluations.

Extrais les notions de cette partie sous forme de fiches :
- titre : nom court de la notion, tel qu'il apparaît dans le cours
- type : definition | propriete | theoreme | methode | exemple | demonstration | remarque
- enonce : ce que dit le cours sur cette notion, fidèlement. Pour une méthode, la démarche.
- importance : 3 = notion centrale qu'un enseignant évaluerait ; 2 = utile (exemples,
  points secondaires) ; 1 = anecdotique (historique, culture générale, détails techniques).
- mots_cles : 2 à 5 mots-clés.

{formules}
N'invente aucune notion absente du texte. Les blocs suivants ont été repérés dans la partie :
chacun doit apparaître dans tes fiches.
{blocs}

Partie « {titre} » :
{texte}"""

_SCIENTIFIQUES = ("math", "physique", "chimie", "informatique")


def regle_formules(matiere: str) -> str:
    """Consigne sur les formules : LaTeX pour les matières scientifiques, texte simple sinon."""
    if any(m in (matiere or "").lower() for m in _SCIENTIFIQUES):
        return ("Formules en LaTeX ($...$). Dans les chaînes JSON, écris chaque barre oblique "
                "LaTeX doublée : \\\\int, \\\\frac, \\\\text.")
    return "Écris en texte simple : pas de LaTeX, pas de symbole $."


_TITRE_NIVEAU1 = re.compile(r"(?m)^#\s+(?!#)(.+?)\s*$")
_TITRE_NIVEAU2 = re.compile(r"(?m)^##\s+(?!#)(.+?)\s*$")
# Pas de détection des titres « 1. Titre » : sur la couche texte d'un PDF LaTeX (Exo7),
# elle confond titres de section, en-têtes de page et sous-questions d'exercices.
TAILLE_MAX_PARTIE = 12000   # au-delà, la partie est coupée (le modèle tronquerait sinon)


def _reperes(cours: str) -> list[tuple[int, int, str]]:
    """(position, numéro, titre) des titres de parties, selon le premier style reconnu."""
    m = list(_PARTIE.finditer(cours))
    if m:
        return [(x.start(), int(x.group(1)), x.group(2).strip(" *#") or f"Partie {x.group(1)}") for x in m]
    for motif in (_TITRE_NIVEAU1, _TITRE_NIVEAU2):
        m = list(motif.finditer(cours))
        if len(m) >= 2:
            return [(x.start(), i, x.group(1).strip(" *#")) for i, x in enumerate(m, 1)]
    return []


def _couper(texte: str, taille: int) -> list[str]:
    """Coupe un texte trop long aux fins de paragraphe, en morceaux d'au plus `taille`."""
    morceaux, courant = [], ""
    for paragraphe in re.split(r"\n\s*\n", texte):
        if courant and len(courant) + len(paragraphe) + 2 > taille:
            morceaux.append(courant)
            courant = ""
        courant = f"{courant}\n\n{paragraphe}" if courant else paragraphe
        while len(courant) > taille:            # paragraphe géant : coupe brute
            morceaux.append(courant[:taille])
            courant = courant[taille:]
    return morceaux + ([courant] if courant.strip() else [])


def decouper_parties(cours: str, taille_max: int = TAILLE_MAX_PARTIE) -> list[dict]:
    """
    [{numero, titre, texte}]. Titres reconnus, dans l'ordre : « Partie N » (style Monka),
    titres Markdown « # » puis « ## » (sortie de l'extraction par vision).
    Sans structure (couche texte brute), le cours est coupé en morceaux. Une partie trop longue est coupée en
    morceaux de même numéro (« titre (suite 2) ») pour ne jamais être tronquée.
    """
    reperes = _reperes(cours)
    brutes = []
    if not reperes:
        brutes = [{"numero": i, "titre": f"Cours (morceau {i})", "texte": t}
                  for i, t in enumerate(_couper(cours, taille_max), 1)]
    else:
        if len(cours[:reperes[0][0]].strip()) > 200:     # introduction avant le 1er titre
            brutes.append({"numero": 0, "titre": "Introduction", "texte": cours[:reperes[0][0]]})
        for i, (debut, numero, titre) in enumerate(reperes):
            fin = reperes[i + 1][0] if i + 1 < len(reperes) else len(cours)
            brutes.append({"numero": numero, "titre": titre, "texte": cours[debut:fin]})
    parties = []
    for p in brutes:
        morceaux = _couper(p["texte"], taille_max) if len(p["texte"]) > taille_max else [p["texte"]]
        for k, texte in enumerate(morceaux, 1):
            titre = p["titre"] if k == 1 else f"{p['titre']} (suite {k})"
            parties.append({"numero": p["numero"], "titre": titre, "texte": texte})
    return parties


def blocs_structurels(texte: str) -> list[str]:
    return [f"- {m.group(1)} : {m.group(2).strip(' *')}" for m in _BLOC.finditer(texte)]


def analyser_cours(cours: str, modele: str, num_ctx: int = 12288,
                   matiere: str = "mathématiques") -> dict:
    """Une partie en échec (modèle qui boucle, délai dépassé) est signalée, les autres continuent."""
    parties = []
    for p in decouper_parties(cours):
        blocs = blocs_structurels(p["texte"])
        entree = {"numero": p["numero"], "titre": p["titre"], "blocs_reperes": len(blocs),
                  "notions": [], "duree_s": 0.0}
        try:
            r = discuter(modele, [{"role": "user", "content": CONSIGNE.format(matiere=matiere, formules=regle_formules(matiere),
                blocs="\n".join(blocs) or "(aucun)", titre=p["titre"], texte=p["texte"][:24000])}],
                schema=SCHEMA_NOTIONS, temperature=0.1, num_ctx=num_ctx, max_tokens=4096,
                timeout=900)
            entree.update(notions=r["json"]["notions"], duree_s=r["duree_s"])
        except (ErreurModele, httpx.HTTPError) as e:
            entree["erreur"] = f"{type(e).__name__} : {e}"
        parties.append(entree)
    return {"parties": parties}


def fiches_pour(analyse: dict, numeros: list[int] | None = None, importance_min: int = 2) -> list[dict]:
    """Notions des parties choisies (toutes si None), filtrées par importance."""
    return [{**n, "partie": p["titre"]} for p in analyse["parties"]
            if numeros is None or p["numero"] in numeros
            for n in p["notions"] if n.get("importance", 1) >= importance_min]
