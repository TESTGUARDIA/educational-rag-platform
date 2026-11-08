"""
Génération de sujets à partir des fiches de notions d'un cours.

Mode A — questions de cours : sur les notions d'importance élevée.
Mode B — problème : sujet en plusieurs questions progressives sur un thème
(ex. « intégrales de Wallis »), avec corrigé et barème, contrôlé en quatre temps :
  1. plan (objectif final, notions mobilisées, progression) ;
  2. rédaction (énoncé, corrigé, barème + vérifications déclarées par le modèle) ;
  3. vérification numérique des affirmations (verification.py) et résolution
     indépendante des questions de calcul, sans le corrigé ;
  4. relecture critique ; correction ciblée des questions en échec (tours_max).
"""

import json
import logging
import time

from .ollama import discuter
from .verification import construire_famille, comparer_valeurs, verifier_probleme, ExpressionRefusee

log = logging.getLogger("testguardai.problemes")

# ─── Schémas de sortie ─────────────────────────────────────────────────────────
SCHEMA_PLAN = {
    "type": "object",
    "properties": {
        "titre": {"type": "string"},
        "objectif_final": {"type": "string"},
        "notions_mobilisees": {"type": "array", "maxItems": 10, "items": {"type": "string"}},
        "etapes": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
            "numero": {"type": "string"}, "intention": {"type": "string"},
            "difficulte": {"type": "integer", "minimum": 1, "maximum": 3}},
            "required": ["numero", "intention", "difficulte"]}},
    },
    "required": ["titre", "objectif_final", "notions_mobilisees", "etapes"],
}

SCHEMA_PROBLEME = {
    "type": "object",
    "properties": {
        "titre": {"type": "string"},
        "introduction": {"type": "string"},
        "definitions": {"type": "array", "maxItems": 3, "items": {"type": "object", "properties": {
            "nom": {"type": "string"}, "parametre": {"type": "string"}, "variable": {"type": "string"},
            "integrande": {"type": "string"}, "borne_inf": {"type": "string"},
            "borne_sup": {"type": "string"}},
            "required": ["nom", "parametre", "variable", "integrande", "borne_inf", "borne_sup"]}},
        "questions": {"type": "array", "maxItems": 8, "items": {"type": "object", "properties": {
            "numero": {"type": "string"},
            "type": {"type": "string", "enum": ["calcul", "demonstration"]},
            "enonce": {"type": "string"}, "corrige": {"type": "string"},
            "reponse_finale_calcul": {"type": "string"},
            "points": {"type": "number"}},
            "required": ["numero", "type", "enonce", "corrige", "reponse_finale_calcul", "points"]}},
        "verifications": {"type": "array", "maxItems": 25, "items": {"type": "object", "properties": {
            "question": {"type": "string"},
            "type": {"type": "string", "enum": ["egalite", "inegalite", "limite"]},
            "description": {"type": "string"},
            "gauche": {"type": "string"}, "droite": {"type": "string"},
            "operateur": {"type": "string", "enum": ["=", "<=", "<", ">=", ">"]},
            "n_min": {"type": "integer"}, "n_max": {"type": "integer"}},
            "required": ["question", "type", "description", "gauche", "droite", "operateur",
                         "n_min", "n_max"]}},
    },
    "required": ["titre", "introduction", "definitions", "questions", "verifications"],
}

SCHEMA_RESOLUTION = {
    "type": "object",
    "properties": {"reponses": {"type": "array", "maxItems": 10, "items": {"type": "object", "properties": {
        "numero": {"type": "string"}, "reponse_finale_calcul": {"type": "string"}},
        "required": ["numero", "reponse_finale_calcul"]}}},
    "required": ["reponses"],
}

SCHEMA_CRITIQUE = {
    "type": "object",
    "properties": {
        "faisable": {"type": "boolean"}, "conforme_au_cours": {"type": "boolean"},
        "progressif": {"type": "boolean"},
        "problemes": {"type": "array", "maxItems": 10, "items": {"type": "object", "properties": {
            "numero": {"type": "string"}, "probleme": {"type": "string"}},
            "required": ["numero", "probleme"]}},
        "note": {"type": "integer", "minimum": 1, "maximum": 5},
        "commentaire": {"type": "string"},
    },
    "required": ["faisable", "conforme_au_cours", "progressif", "problemes", "note", "commentaire"],
}

SCHEMA_QUESTIONS_COURS = {
    "type": "object",
    "properties": {"questions": {"type": "array", "maxItems": 15, "items": {"type": "object", "properties": {
        "notion": {"type": "string"},
        "type_question": {"type": "string",
                          "enum": ["definition", "enonce", "justification", "application"]},
        "question": {"type": "string"}, "reponse": {"type": "string"}},
        "required": ["notion", "type_question", "question", "reponse"]}}},
    "required": ["questions"],
}

# ─── Consignes ─────────────────────────────────────────────────────────────────
SYNTAXE = """Syntaxe des champs integrande, borne_inf, borne_sup, gauche, droite, reponse_finale_calcul :
expressions SymPy en texte brut (PAS de LaTeX) : ^ pour la puissance, pi, e, sqrt(), ln(), exp(),
sin(), cos(), factorial(), binomial(), la variable entière n, et les familles d'intégrales déclarées
dans "definitions", appelées par leur nom : W(n), W(n+2), W(2*n).
Exemple pour une autre famille, J_n = intégrale de 0 à 1 de x^n e^x dx :
  "definitions": [{"nom": "J", "parametre": "n", "variable": "x", "integrande": "x^n*exp(x)",
                   "borne_inf": "0", "borne_sup": "1"}]
  "verifications": [
    {"question": "1", "type": "egalite", "description": "valeur de J_0", "gauche": "J(0)",
     "droite": "e-1", "operateur": "=", "n_min": 0, "n_max": 0},
    {"question": "2", "type": "egalite", "description": "relation de récurrence",
     "gauche": "J(n+1)", "droite": "e-(n+1)*J(n)", "operateur": "=", "n_min": 0, "n_max": 10},
    {"question": "3", "type": "inegalite", "description": "suite décroissante",
     "gauche": "J(n+1)", "droite": "J(n)", "operateur": "<=", "n_min": 0, "n_max": 15},
    {"question": "4", "type": "limite", "description": "limite de J_n", "gauche": "J(n)",
     "droite": "0", "operateur": "=", "n_min": 0, "n_max": 0}]"""

CONSIGNE_PLAN = """Tu es professeur de mathématiques ({niveau}) et tu conçois un problème d'évaluation.

Thème demandé : {theme}
Difficulté : {difficulte}/3

Notions du cours disponibles (n'utilise QUE celles-ci, plus les prérequis normaux du niveau) :
{fiches}

Conçois le plan du problème : un objectif final ambitieux mais accessible (ex. un résultat
classique du thème), les notions du cours mobilisées, et 4 à 7 étapes progressives
(1a, 1b, 2a…), chacune s'appuyant sur les précédentes, de la plus simple à la plus difficile."""

CONSIGNE_REDACTION = """Tu es professeur de mathématiques ({niveau}). Rédige le problème complet
correspondant à ce plan :
{plan}

Notions du cours disponibles (n'utilise QUE celles-ci, plus les prérequis normaux du niveau) :
{fiches}

Exigences :
- "introduction" : le contexte et les définitions, en LaTeX ($...$).
- Une question par étape du plan. "enonce" en LaTeX, autonome et sans ambiguïté.
- "corrige" : solution détaillée et rigoureuse, étape par étape, en LaTeX.
- "type" : "calcul" si la réponse est une valeur ou une expression, "demonstration" sinon.
- "reponse_finale_calcul" : pour une question de calcul, le résultat en syntaxe SymPy
  (ex. "pi/2", "(n+1)/(n+2)*W(n)") ; chaîne vide pour une démonstration.
- "points" : barème, total sur 20.
- "definitions" : chaque famille d'intégrales indexée par n utilisée dans le problème.
- "verifications" : pour CHAQUE résultat affirmé dans le corrigé (valeur, relation,
  inégalité, limite), une vérification qu'un programme contrôlera numériquement.
  Au moins une par question quand c'est possible.

{syntaxe}"""

CONSIGNE_CORRECTION = """Le problème suivant contient des erreurs détectées automatiquement.

Problème (JSON) :
{probleme}

Erreurs détectées :
{erreurs}

Corrige le problème : modifie les questions concernées (énoncé, corrigé, résultat) ou les
vérifications mal écrites, sans changer ce qui est juste. Une vérification « faux » signifie que
l'affirmation est fausse numériquement : c'est le corrigé (ou l'énoncé) qu'il faut corriger, pas
la vérification. Renvoie le problème complet.

{syntaxe}"""

CONSIGNE_RESOLUTION = """Résous ce problème de mathématiques. Pour chaque question de calcul,
donne uniquement le résultat final en syntaxe SymPy (texte brut, pas de LaTeX : ^, pi, e, sqrt(),
ln(), factorial(), et les familles {familles} appelées par leur nom, ex. W(n)).

{enonce}"""

CONSIGNE_CRITIQUE = """Tu es inspecteur de mathématiques. Évalue ce sujet destiné à des élèves
de {niveau}, à partir de ce cours :
{fiches}

Sujet :
{enonce}

Critères : faisable (chaque question est résoluble avec ce qui précède), conforme_au_cours
(n'exige pas de notion hors programme ou absente du cours), progressif (difficulté croissante).
Liste les problèmes précis question par question. Note de 1 à 5."""

CONSIGNE_QUESTIONS_COURS = """Tu es professeur de mathématiques ({niveau}). Rédige {nb} questions
de cours portant sur les notions importantes ci-dessous (une notion par question, en privilégiant
les plus centrales). Varie les types : definition, enonce (énoncer un théorème), justification
(démontrer ou expliquer une propriété du cours), application (utiliser une méthode sur un cas simple).
Réponses complètes et exactes, conformes au cours, formules en LaTeX ($...$).

Notions :
{fiches}"""


def _fiches_texte(fiches: list[dict], limite: int = 9000) -> str:
    lignes = [f"- [{f['type']}] {f['titre']} ({f.get('partie', '')}) : {f['enonce']}" for f in fiches]
    texte, total = [], 0
    for l in lignes:
        if total + len(l) > limite:
            break
        texte.append(l)
        total += len(l)
    return "\n".join(texte)


def enonce_eleve(probleme: dict) -> str:
    qs = "\n\n".join(f"**{q['numero']}.** {q['enonce']}" for q in probleme.get("questions", []))
    return f"# {probleme.get('titre', '')}\n\n{probleme.get('introduction', '')}\n\n{qs}"


def _budget(raisonnement) -> int:
    """Tokens de sortie autorisés : la réflexion (mode raisonnement) en consomme beaucoup."""
    return 12000 if raisonnement not in (None, False) else 6000


# ─── Mode B : problème ─────────────────────────────────────────────────────────
def _familles(probleme: dict) -> dict:
    familles = {}
    for d in probleme.get("definitions", []):
        try:
            f = construire_famille(d)
            familles[f.nom] = f
        except (ExpressionRefusee, KeyError):
            pass
    return familles


def resoudre_independamment(probleme: dict, modele: str, raisonnement, num_ctx: int) -> dict:
    calculs = [q for q in probleme.get("questions", []) if q.get("type") == "calcul"
               and q.get("reponse_finale_calcul", "").strip()]
    if not calculs:
        return {"comparaisons": [], "duree_s": 0}
    familles = _familles(probleme)
    r = discuter(modele, [{"role": "user", "content": CONSIGNE_RESOLUTION.format(
        familles=", ".join(familles) or "(aucune)", enonce=enonce_eleve(probleme))}],
        schema=SCHEMA_RESOLUTION, raisonnement=raisonnement, temperature=0.0, num_ctx=num_ctx,
        max_tokens=_budget(raisonnement))
    reponses = {str(x["numero"]).strip(): x["reponse_finale_calcul"] for x in r["json"]["reponses"]}
    comparaisons = []
    for q in calculs:
        autre = reponses.get(str(q["numero"]).strip(), "")
        accord = comparer_valeurs(q["reponse_finale_calcul"], autre, familles) if autre else None
        comparaisons.append({"question": q["numero"], "corrige": q["reponse_finale_calcul"],
                             "resolution": autre,
                             "statut": {True: "concordant", False: "discordant",
                                        None: "non comparable"}[accord]})
    return {"comparaisons": comparaisons, "duree_s": r["duree_s"]}


def _erreurs(verif: dict, resolution: dict) -> list[str]:
    erreurs = [f"- Définition invalide : {e}" for e in verif["erreurs_definitions"]]
    for r in verif["resultats"]:
        if r["statut"] != "ok":
            erreurs.append(f"- Question {r['question']}, vérification « {r['description']} » "
                           f"({r['gauche']} vs {r['droite']}) : {r['statut']} — {'; '.join(r['detail'])}")
    for c in resolution["comparaisons"]:
        if c["statut"] == "discordant":
            erreurs.append(f"- Question {c['question']} : le corrigé donne {c['corrige']} mais une "
                           f"résolution indépendante trouve {c['resolution']}. Vérifie le calcul.")
    return erreurs


def generer_probleme(fiches: list[dict], theme: str, modele: str, *, raisonnement=True,
                     difficulte: int = 2, niveau: str = "Terminale, spécialité mathématiques",
                     tours_max: int = 2, num_ctx: int = 16384) -> dict:
    t0 = time.monotonic()
    texte_fiches = _fiches_texte(fiches)
    etapes = []

    r = discuter(modele, [{"role": "user", "content": CONSIGNE_PLAN.format(
        niveau=niveau, theme=theme, difficulte=difficulte, fiches=texte_fiches)}],
        schema=SCHEMA_PLAN, raisonnement=raisonnement, temperature=0.4, num_ctx=num_ctx,
        max_tokens=_budget(raisonnement))
    plan = r["json"]
    etapes.append({"etape": "plan", "duree_s": r["duree_s"]})
    log.warning("Plan : %s (%.0f s)", plan["titre"], r["duree_s"])

    r = discuter(modele, [{"role": "user", "content": CONSIGNE_REDACTION.format(
        niveau=niveau, plan=json.dumps(plan, ensure_ascii=False, indent=1), fiches=texte_fiches,
        syntaxe=SYNTAXE)}], schema=SCHEMA_PROBLEME, raisonnement=raisonnement,
        temperature=0.3, num_ctx=num_ctx, max_tokens=_budget(raisonnement))
    probleme = r["json"]
    etapes.append({"etape": "redaction", "duree_s": r["duree_s"]})
    log.warning("Rédaction : %d questions (%.0f s)", len(probleme["questions"]), r["duree_s"])

    historique = []
    for tour in range(tours_max + 1):
        verif = verifier_probleme(probleme)
        resolution = resoudre_independamment(probleme, modele, raisonnement, num_ctx)
        erreurs = _erreurs(verif, resolution)
        historique.append({"tour": tour, "verifications_ok": verif["ok"], "faux": verif["faux"],
                           "erreurs_verif": verif["erreur"],
                           "discordances": sum(c["statut"] == "discordant"
                                               for c in resolution["comparaisons"]),
                           "duree_resolution_s": resolution["duree_s"]})
        log.warning("Tour %d : %d vérif. ok, %d fausses, %d illisibles, %d discordances", tour,
                    verif["ok"], verif["faux"], verif["erreur"], historique[-1]["discordances"])
        if not erreurs or tour == tours_max:
            break
        r = discuter(modele, [{"role": "user", "content": CONSIGNE_CORRECTION.format(
            probleme=json.dumps(probleme, ensure_ascii=False), erreurs="\n".join(erreurs),
            syntaxe=SYNTAXE)}], schema=SCHEMA_PROBLEME, raisonnement=raisonnement,
            temperature=0.2, num_ctx=num_ctx, max_tokens=_budget(raisonnement))
        probleme = r["json"]
        etapes.append({"etape": f"correction_{tour + 1}", "duree_s": r["duree_s"]})

    r = discuter(modele, [{"role": "user", "content": CONSIGNE_CRITIQUE.format(
        niveau=niveau, fiches=texte_fiches, enonce=enonce_eleve(probleme))}],
        schema=SCHEMA_CRITIQUE, raisonnement=raisonnement, temperature=0.0, num_ctx=num_ctx,
        max_tokens=_budget(raisonnement))
    etapes.append({"etape": "critique", "duree_s": r["duree_s"]})

    return {"theme": theme, "modele": modele, "raisonnement": raisonnement, "plan": plan,
            "probleme": probleme, "verification": verif, "resolution": resolution,
            "critique": r["json"], "historique": historique, "etapes": etapes,
            "erreurs_restantes": erreurs, "duree_totale_s": round(time.monotonic() - t0, 1)}


# ─── Mode A : questions de cours ───────────────────────────────────────────────
def generer_questions_cours(fiches: list[dict], nb: int, modele: str, *,
                            niveau: str = "Terminale, spécialité mathématiques",
                            num_ctx: int = 12288) -> dict:
    importantes = sorted(fiches, key=lambda f: -f.get("importance", 1))
    r = discuter(modele, [{"role": "user", "content": CONSIGNE_QUESTIONS_COURS.format(
        niveau=niveau, nb=nb, fiches=_fiches_texte(importantes))}],
        schema=SCHEMA_QUESTIONS_COURS, temperature=0.2, num_ctx=num_ctx)
    titres = {f["titre"].lower() for f in fiches}
    questions = r["json"]["questions"][:nb]
    for q in questions:   # traçabilité : la notion citée existe-t-elle dans le cours ?
        q["notion_trouvee_dans_le_cours"] = q["notion"].lower() in titres
    return {"questions": questions, "duree_s": r["duree_s"], "modele": modele}


# ─── Rendu ─────────────────────────────────────────────────────────────────────
def rendre_markdown(res: dict) -> str:
    p, v, c = res["probleme"], res["verification"], res["critique"]
    lignes = [enonce_eleve(p), "\n---\n# Corrigé et barème\n"]
    for q in p["questions"]:
        lignes.append(f"**{q['numero']}.** ({q['points']} pts)\n\n{q['corrige']}\n")
    lignes.append("\n---\n# Rapport de génération\n")
    lignes.append(f"- Modèle : `{res['modele']}`, raisonnement : `{res['raisonnement']}`, "
                  f"durée totale : {res['duree_totale_s']:.0f} s")
    lignes.append(f"- Étapes : " + ", ".join(f"{e['etape']} {e['duree_s']:.0f} s" for e in res["etapes"]))
    lignes.append(f"- Vérifications numériques : **{v['ok']} ok**, {v['faux']} fausses, "
                  f"{v['erreur']} illisibles (familles : {', '.join(v['familles']) or 'aucune'})")
    for r in v["resultats"]:
        marque = {"ok": "✅", "faux": "❌", "erreur": "⚠️"}[r["statut"]]
        lignes.append(f"  - {marque} Q{r['question']} — {r['description']} : `{r['gauche']}` "
                      f"vs `{r['droite']}` {('— ' + '; '.join(r['detail'])) if r['statut'] != 'ok' else ''}")
    for comp in res["resolution"]["comparaisons"]:
        lignes.append(f"  - Résolution indépendante Q{comp['question']} : {comp['statut']} "
                      f"(corrigé `{comp['corrige']}`, résolution `{comp['resolution']}`)")
    lignes.append(f"- Tours de correction : " + " → ".join(
        f"[{h['faux']} fausses, {h['erreurs_verif']} illisibles, {h['discordances']} discordances]"
        for h in res["historique"]))
    lignes.append(f"- Relecture critique : note {c['note']}/5, faisable={c['faisable']}, "
                  f"conforme={c['conforme_au_cours']}, progressif={c['progressif']}")
    lignes += [f"  - Q{x['numero']} : {x['probleme']}" for x in c["problemes"]]
    lignes.append(f"  - {c['commentaire']}")
    return "\n".join(lignes)
