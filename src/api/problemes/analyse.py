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

CONSIGNE = """Tu analyses une partie d'un cours de mathématiques pour préparer des évaluations.

Extrais les notions de cette partie sous forme de fiches :
- titre : nom court de la notion (ex. « Intégration par parties »)
- type : definition | propriete | theoreme | methode | exemple | demonstration | remarque
- enonce : l'énoncé exact du cours, formules en LaTeX ($...$). Pour une méthode, la démarche.
- importance : 3 = notion centrale qu'un enseignant évaluerait (définitions, théorèmes,
  méthodes clés) ; 2 = utile (exemples types, propriétés secondaires) ; 1 = anecdotique
  (historique, culture générale, détails de calculatrice).
- mots_cles : 2 à 5 mots-clés.

N'invente aucune notion absente du texte. Les blocs suivants ont été repérés dans la partie :
chacun doit apparaître dans tes fiches.
{blocs}

Partie « {titre} » :
{texte}"""


def decouper_parties(cours: str) -> list[dict]:
    """[{numero, titre, texte}] ; une seule partie « Cours » si aucun titre n'est trouvé."""
    reperes = list(_PARTIE.finditer(cours))
    if not reperes:
        return [{"numero": 1, "titre": "Cours", "texte": cours}]
    parties = []
    for i, m in enumerate(reperes):
        fin = reperes[i + 1].start() if i + 1 < len(reperes) else len(cours)
        parties.append({"numero": int(m.group(1)), "titre": m.group(2).strip(" *#") or f"Partie {m.group(1)}",
                        "texte": cours[m.start():fin]})
    return parties


def blocs_structurels(texte: str) -> list[str]:
    return [f"- {m.group(1)} : {m.group(2).strip(' *')}" for m in _BLOC.finditer(texte)]


def analyser_cours(cours: str, modele: str, num_ctx: int = 12288) -> dict:
    """Une partie en échec (modèle qui boucle, délai dépassé) est signalée, les autres continuent."""
    parties = []
    for p in decouper_parties(cours):
        blocs = blocs_structurels(p["texte"])
        entree = {"numero": p["numero"], "titre": p["titre"], "blocs_reperes": len(blocs),
                  "notions": [], "duree_s": 0.0}
        try:
            r = discuter(modele, [{"role": "user", "content": CONSIGNE.format(
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
