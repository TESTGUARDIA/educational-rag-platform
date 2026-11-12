"""Banc de mesure du pôle Génération (src/generation/banc), modèles simulés."""

import csv
import json

import pytest

from src.generation.banc import banc

ANALYSE = {"parties": [{"numero": 1, "titre": "Intégrales", "blocs_reperes": 2, "duree_s": 1.0,
                        "notions": [{"titre": "IPP", "type": "theoreme", "enonce": "...",
                                     "importance": 3, "mots_cles": []}]}]}


def _probleme(modele):
    return {"probleme": {"titre": "Wallis", "introduction": "...", "definitions": [], "verifications": [],
                         "questions": [{"numero": "1", "type": "calcul", "enonce": "Q", "corrige": "C",
                                        "reponse_finale_calcul": "1", "points": 20}]},
            "verification": {"ok": 3, "faux": 1, "erreur": 0, "familles": [], "resultats": [],
                             "erreurs_definitions": []},
            "resolution": {"comparaisons": [], "duree_s": 0}, "critique": {
                "note": 4, "faisable": True, "conforme_au_cours": True, "progressif": True,
                "problemes": [], "commentaire": "ok"},
            "historique": [{"tour": 0}], "etapes": [], "erreurs_restantes": [],
            "duree_totale_s": 60.0, "modele": modele, "raisonnement": True, "theme": "Wallis", "plan": {}}


@pytest.fixture()
def faux(monkeypatch, tmp_path):
    pdf = tmp_path / "cours.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(banc.ex, "texte_brut", lambda p, pages=None: "# A\n$x$ texte\n# B\n$y$")
    monkeypatch.setattr(banc.ex, "extraire_cours", lambda *a, **k: "# A\n$x$ vision\n# B\n$y$")
    monkeypatch.setattr(banc.an, "analyser_cours", lambda texte, modele, **k: ANALYSE)

    def questions(fiches, nb, modele, **k):
        if modele == "en-panne":
            raise RuntimeError("Ollama injoignable")
        return {"duree_s": 10.0, "modele": modele, "questions": [
            {"notion": "IPP", "type_question": "enonce", "question": f"Énoncer l'IPP ({modele})",
             "reponse": "...", "notion_trouvee_dans_le_cours": True},
            {"notion": "IPP", "type_question": "enonce", "question": f"Énoncer l'IPP ({modele}).",
             "reponse": "...", "notion_trouvee_dans_le_cours": False}]}
    monkeypatch.setattr(banc.gen, "generer_questions_cours", questions)
    monkeypatch.setattr(banc.gen, "generer_probleme", lambda fiches, theme, modele, **k: _probleme(modele))
    monkeypatch.setattr(banc.gen, "rendre_markdown",
                        lambda res: f"# Sujet {res['modele']}\n\n---\n# Rapport de génération\nsecret")
    cas = [{"id": "q", "pdf": str(pdf), "mode": "questions", "nb": 2, "matiere": "maths"},
           {"id": "p", "pdf": str(pdf), "mode": "probleme", "theme": "Wallis"},
           {"id": "absent", "pdf_env": "VARIABLE_QUI_N_EXISTE_PAS", "mode": "questions"}]
    configs = [{"nom": "modele-a", "modele": "a", "extraction": "texte"},
               {"nom": "modele-b", "modele": "b", "extraction": "vision", "modele_vision": "v"},
               {"nom": "panne", "modele": "en-panne", "extraction": "texte"}]
    return cas, configs, tmp_path


def test_lancer_puis_resumer(faux, monkeypatch):
    cas, configs, tmp = faux
    monkeypatch.setattr(banc, "RACINE", tmp)
    run = banc.lancer(cas, configs, sortie=tmp / "sortie_banc")

    mesures = json.loads((run / "metriques.json").read_text(encoding="utf-8"))
    assert len(mesures) == 9
    statut = {(m["config"], m["cas"]): m["statut"] for m in mesures}
    assert statut[("modele-a", "absent")].startswith("ignoré")
    assert statut[("panne", "q")] == "erreur"                 # la panne n'arrête pas le banc
    assert statut[("panne", "p")] == "ok"
    qa = next(m for m in mesures if m["config"] == "modele-a" and m["cas"] == "q")
    assert qa["quasi_doublons"] == 1 and qa["notion_trouvee_dans_le_cours_pct"] == 50.0
    assert qa["extraction"] == "texte" and qa["formules_latex"] == 2

    # Notation à l'aveugle : aucun nom de configuration ni rapport dans a_noter.md
    a_noter = (run / "a_noter.md").read_text(encoding="utf-8")
    assert "modele-a" not in a_noter and "modele-b" not in a_noter and "secret" not in a_noter
    with open(run / "notation.csv", encoding="utf-8-sig") as f:
        lignes = list(csv.DictReader(f, delimiter=";"))
    correspondance = json.loads((run / "correspondance.json").read_text(encoding="utf-8"))
    assert len(lignes) == len(correspondance) == 7            # 2×2 questions + 3 problèmes
    assert len({l["code"] for l in lignes}) == 7

    # L'enseignant remplit la grille : modele-a mieux noté que modele-b
    for l in lignes:
        bon = correspondance[l["code"]]["config"] == "modele-a"
        for c in banc.CRITERES:
            l[c] = "2" if bon else "1"
        l["utilisable"] = "oui" if bon else "non"
    with open(run / "notation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=banc.COLONNES_NOTATION, delimiter=";")
        w.writeheader()
        w.writerows(lignes)

    synthese = banc.resumer(run).read_text(encoding="utf-8")
    with open(run / "synthese.csv", encoding="utf-8-sig") as f:
        par_config = {l["config"]: l for l in csv.DictReader(f, delimiter=";")}
    assert par_config["modele-a"]["note_enseignant_sur_10"] == "10.0"
    assert par_config["modele-b"]["note_enseignant_sur_10"] == "5.0"
    assert par_config["modele-a"]["acceptation_pct"] == "100.0"
    assert par_config["modele-b"]["rejet_pct"] == "100.0"
    assert par_config["modele-a"]["pb_taux_verif_ok_pct"] == "75.0"
    assert "Ollama injoignable" in synthese                    # cas non aboutis listés


def test_quasi_doublons():
    assert banc.quasi_doublons(["Énoncer le théorème.", "Énoncer le théorème !", "Calculer I_0."]) == 1


def test_selectionner():
    elements = [{"nom": "a", "par_defaut": True}, {"nom": "b"}]
    assert [e["nom"] for e in banc.selectionner(elements, "nom", None, lambda e: e.get("par_defaut"))] == ["a"]
    assert [e["nom"] for e in banc.selectionner(elements, "nom", "b")] == ["b"]
    with pytest.raises(ValueError):
        banc.selectionner(elements, "nom", "z")


def test_fichiers_du_banc_coherents():
    cas = banc.charger_json("cas.json")
    configs = banc.charger_json("configs.json")
    assert len({c["id"] for c in cas}) == len(cas)
    assert all(c["mode"] in ("questions", "probleme") for c in cas)
    assert all(c.get("theme") for c in cas if c["mode"] == "probleme")
    assert all(c.get("modele_vision") for c in configs if c.get("extraction") == "vision")
