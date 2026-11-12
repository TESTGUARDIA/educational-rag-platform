"""Circuit de génération de problèmes (src/generation/problemes/generation.py), modèle simulé."""

from src.generation.problemes import generation

DEF_W = {"nom": "W", "parametre": "n", "variable": "x", "integrande": "sin(x)^n",
         "borne_inf": "0", "borne_sup": "pi/2"}


def _probleme(recurrence: str, w1: str) -> dict:
    return {
        "titre": "Intégrales de Wallis",
        "introduction": "Pour tout entier $n$, $W_n=\\int_0^{\\pi/2}\\sin^n(x)\\,dx$.",
        "definitions": [DEF_W],
        "questions": [
            {"numero": "1", "type": "calcul", "enonce": "Calculer $W_1$.", "corrige": "...",
             "reponse_finale_calcul": w1, "points": 4},
            {"numero": "2", "type": "demonstration", "enonce": "Montrer la récurrence.",
             "corrige": "...", "reponse_finale_calcul": "", "points": 16},
        ],
        "verifications": [
            {"question": "1", "type": "egalite", "description": "W_1", "gauche": "W(1)",
             "droite": w1, "operateur": "=", "n_min": 0, "n_max": 0},
            {"question": "2", "type": "egalite", "description": "récurrence", "gauche": "W(n+2)",
             "droite": recurrence, "operateur": "=", "n_min": 0, "n_max": 8},
        ],
    }


class FauxModele:
    """Rejoue des réponses selon le schéma demandé ; garde la trace des consignes reçues."""

    def __init__(self):
        self.consignes = []
        self.redactions = [_probleme("n/(n+2)*W(n)", "2"),          # deux erreurs
                           _probleme("(n+1)/(n+2)*W(n)", "1")]       # corrigé juste

    def __call__(self, modele, messages, *, schema=None, **kw):
        self.consignes.append(messages[-1]["content"])
        s = schema
        if s is generation.SCHEMA_PLAN:
            j = {"titre": "Wallis", "objectif_final": "récurrence", "notions_mobilisees": ["IPP"],
                 "etapes": [{"numero": "1", "intention": "W_1", "difficulte": 1}]}
        elif s is generation.SCHEMA_PROBLEME:
            j = self.redactions.pop(0)
        elif s is generation.SCHEMA_RESOLUTION:
            j = {"reponses": [{"numero": "1", "reponse_finale_calcul": "1"}]}
        elif s is generation.SCHEMA_CRITIQUE:
            j = {"faisable": True, "conforme_au_cours": True, "progressif": True,
                 "problemes": [], "note": 4, "commentaire": "ok"}
        else:
            raise AssertionError(f"schéma inattendu : {s}")
        return {"contenu": "", "json": j, "pensee": "", "duree_s": 0.0, "tokens": 0}


def test_boucle_de_correction(monkeypatch):
    faux = FauxModele()
    monkeypatch.setattr(generation, "discuter", faux)
    fiches = [{"titre": "Intégration par parties", "type": "theoreme", "enonce": "...",
               "importance": 3, "partie": "Partie 5"}]
    res = generation.generer_probleme(fiches, "intégrales de Wallis", "faux", tours_max=2)

    premier, second = res["historique"]
    assert premier["faux"] == 2 and premier["discordances"] == 1      # erreurs détectées
    assert second["faux"] == 0 and second["discordances"] == 0        # corrigées au tour suivant
    assert res["erreurs_restantes"] == []
    assert [e["etape"] for e in res["etapes"]] == ["plan", "redaction", "correction_1", "critique"]

    consigne_correction = next(c for c in faux.consignes if "erreurs détectées" in c)
    assert "récurrence" in consigne_correction and "résolution indépendante trouve 1" in consigne_correction

    md = generation.rendre_markdown(res)
    assert "Intégrales de Wallis" in md and "2 ok" in md


def test_questions_de_cours_tracabilite(monkeypatch):
    def faux(modele, messages, **kw):
        return {"json": {"questions": [
            {"notion": "Intégration par parties", "type_question": "enonce", "question": "Q",
             "reponse": "R"},
            {"notion": "Notion inventée", "type_question": "definition", "question": "Q",
             "reponse": "R"}]}, "duree_s": 0.0}
    monkeypatch.setattr(generation, "discuter", faux)
    fiches = [{"titre": "Intégration par parties", "type": "theoreme", "enonce": "...", "importance": 3}]
    qs = generation.generer_questions_cours(fiches, 2, "faux")["questions"]
    assert [q["notion_trouvee_dans_le_cours"] for q in qs] == [True, False]


def test_questions_qui_recopient_leur_reponse_ecartees(monkeypatch):
    # Sortie réelle de llama3.2:3b sur la photosynthèse : la « question » recopie la réponse
    def faux(modele, messages, **kw):
        return {"json": {"questions": [
            {"notion": "Stroma", "type_question": "definition",
             "question": "$\text{Le stroma est le fluide aqueux du chloroplaste.}$",
             "reponse": "\text{Le stroma est le fluide aqueux du chloroplaste.}"},
            {"notion": "Stroma", "type_question": "definition", "question": "Définir le stroma.",
             "reponse": "Le stroma est le fluide aqueux situé à l'intérieur du chloroplaste."}]},
            "duree_s": 0.0}
    monkeypatch.setattr(generation, "discuter", faux)
    res = generation.generer_questions_cours(
        [{"titre": "Stroma", "type": "definition", "enonce": "...", "importance": 3}], 5, "faux",
        matiere="SVT")
    assert [q["question"] for q in res["questions"]] == ["Définir le stroma."]
    assert res["ecartees_question_egale_reponse"] == 1


def test_regle_formules_selon_la_matiere():
    from src.generation.problemes.analyse import CONSIGNE, regle_formules
    assert "LaTeX" in regle_formules("mathématiques") and "LaTeX" in regle_formules("physique-chimie")
    assert "pas de LaTeX" in regle_formules("SVT") and "pas de LaTeX" in regle_formules("histoire")
    assert "Intégration par parties" not in CONSIGNE          # plus d'exemple propre aux maths
    consigne_svt = generation.CONSIGNE_QUESTIONS_COURS.format(
        matiere="SVT", niveau="lycée", nb=3, fiches="", formules=regle_formules("SVT"))
    assert "\int" not in consigne_svt and "ne contient JAMAIS" in consigne_svt
