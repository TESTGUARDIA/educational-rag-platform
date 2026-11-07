"""
J1 — parcours complet : dépôt d'un cours, génération, réponse d'un élève,
correction automatique, détection IA, notation par l'enseignant.
"""

import sqlite3
import sys
import types

import pytest

from src.web.back import main, services

from .conftest import PDF


def _creer_devoir(client, prof, titre="Devoir test"):
    r = client.post("/devoirs", json={"titre": titre}, headers=prof)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _generer(client, prof, did, nb=3, **kw):
    return client.post(f"/devoirs/{did}/generer", headers=prof,
                       files={"fichier": ("cours.pdf", PDF, "application/pdf")},
                       data={"nb_questions": str(nb)}, **kw)


# ─── Parcours complet ──────────────────────────────────────────────────────────
def test_parcours_complet(client, prof, eleve, faux_modeles):
    did = _creer_devoir(client, prof)

    # 1. Génération depuis le PDF
    r = _generer(client, prof, did, nb=3)
    assert r.status_code == 200, r.text
    corps = r.json()
    assert corps["nb_generees"] == 3
    assert [q["numero_ordre"] for q in corps["questions"]] == [1, 2, 3]
    assert corps["questions"][0]["reponse_attendue"] == "Réponse 1"
    assert faux_modeles["generer"] == [("cours.pdf", 3)]

    # 2. L'élève voit les questions et rend sa copie
    qs = client.get(f"/devoirs/{did}/questions", headers=eleve).json()
    reponses = [{"question_id": qs[0]["id"], "texte": "Réponse 1"},        # exacte : 1.0
                {"question_id": qs[1]["id"], "texte": "autre chose"},      # 0.5
                {"question_id": qs[2]["id"], "texte": "autre chose"}]      # 0.5
    r = client.post("/soumissions", json={"devoir_id": did, "reponses": reponses}, headers=eleve)
    assert r.status_code == 200, r.text
    assert r.json()["statut_analyse"] == "en_cours"
    sid = r.json()["id"]

    # 3. La détection IA a tourné en arrière-plan, avec le seuil du modèle
    assert faux_modeles["detecter"] == [["Réponse 1", "autre chose", "autre chose"]]
    sub = next(s for s in client.get("/soumissions", headers=prof).json() if s["id"] == sid)
    assert sub["ia"]["statut_analyse"] == "analysé"
    assert sub["ia"]["score_ia_global"] == 0.7
    assert sub["ia"]["seuil"] == 0.798

    # 4. Correction automatique : (1 + 0.5 + 0.5) / 3 × 20 = 13.33 -> 13.5
    r = client.post(f"/soumissions/{sid}/corriger-auto", headers=prof)
    assert r.status_code == 200, r.text
    assert r.json()["note_ia_suggeree"] == 13.5
    assert len(r.json()["details"]) == 3

    # 5. L'enseignant valide la note
    r = client.post(f"/soumissions/{sid}/noter", json={"note": 14}, headers=prof)
    assert r.status_code == 200
    sub = next(s for s in client.get("/soumissions", headers=prof).json() if s["id"] == sid)
    assert sub["note_finale_humaine"] == 14
    assert sub["correction"]["statut_correction"] == "validé"


def test_stats_utilisent_le_seuil_du_modele(client, prof, eleve, faux_modeles):
    avant = client.get("/stats", headers=prof).json()["detected"]
    did = _creer_devoir(client, prof)
    _generer(client, prof, did, nb=1)
    qid = client.get(f"/devoirs/{did}/questions", headers=prof).json()[0]["id"]
    client.post("/soumissions", json={"devoir_id": did, "reponses": [
        {"question_id": qid, "texte": "texte"}]}, headers=eleve)
    # score 0.7 : au-dessus de 0.5 mais sous le seuil du modèle (0.798) -> non compté
    assert client.get("/stats", headers=prof).json()["detected"] == avant


# ─── Génération ────────────────────────────────────────────────────────────────
def test_generer_depuis_document_deja_depose_et_numerotation(client, prof, faux_modeles):
    r = client.post("/documents", headers=prof,
                    files=[("files", ("chap1.pdf", PDF, "application/pdf"))])
    doc_id = r.json()["sauvegardes"][0]["id"]
    did = _creer_devoir(client, prof)

    _generer(client, prof, did, nb=2)
    r = client.post(f"/devoirs/{did}/generer", headers=prof,
                    data={"document_id": str(doc_id), "nb_questions": "2"})
    assert r.status_code == 200, r.text
    assert [q["numero_ordre"] for q in r.json()["questions"]] == [1, 2, 3, 4]
    assert faux_modeles["generer"][-1] == ("chap1.pdf", 2)


@pytest.mark.parametrize("kwargs, code", [
    ({"files": {"fichier": ("x.pdf", b"pas un pdf", "application/pdf")}}, 400),
    ({"data": {"nb_questions": "3"}}, 400),                                   # ni fichier ni document
    ({"files": {"fichier": ("x.pdf", PDF, "application/pdf")}, "data": {"nb_questions": "0"}}, 400),
    ({"data": {"document_id": "9999"}}, 404),
])
def test_generer_entrees_invalides(client, prof, faux_modeles, kwargs, code):
    did = _creer_devoir(client, prof)
    r = client.post(f"/devoirs/{did}/generer", headers=prof, **kwargs)
    assert r.status_code == code, r.text
    assert faux_modeles["generer"] == []


def test_generer_reserve_au_proprietaire(client, prof, eleve, faux_modeles):
    did = _creer_devoir(client, prof)
    assert _generer(client, eleve, did).status_code == 403
    assert _generer(client, prof, 9999).status_code == 404


def test_generer_echec_du_modele(client, prof, monkeypatch):
    did = _creer_devoir(client, prof)

    def panne(*a):
        raise ConnectionError("Ollama injoignable")
    monkeypatch.setattr(services, "generer_questions", panne)
    r = _generer(client, prof, did)
    assert r.status_code == 502 and "Ollama injoignable" in r.json()["detail"]

    monkeypatch.setattr(services, "generer_questions", lambda *a: [])
    assert _generer(client, prof, did).status_code == 502
    assert client.get(f"/devoirs/{did}/questions", headers=prof).json() == []


# ─── Correction ────────────────────────────────────────────────────────────────
def test_corriger_ignore_les_questions_sans_reponse_attendue(client, prof, eleve,
                                                              faux_modeles, monkeypatch):
    monkeypatch.setattr(services, "generer_questions", lambda *a: [
        {"question": "Q1 ?", "reponse": "R1"}, {"question": "Q2 ?", "reponse": ""}])
    did = _creer_devoir(client, prof)
    _generer(client, prof, did)
    qs = client.get(f"/devoirs/{did}/questions", headers=prof).json()
    sid = client.post("/soumissions", headers=eleve, json={"devoir_id": did, "reponses": [
        {"question_id": qs[0]["id"], "texte": "R1"},
        {"question_id": qs[1]["id"], "texte": "n'importe"}]}).json()["id"]

    r = client.post(f"/soumissions/{sid}/corriger-auto", headers=prof).json()
    assert r["note_ia_suggeree"] == 20.0                 # seule Q1 est notée
    assert r["details"][1]["statut"] == "non_corrigée"
    assert len(faux_modeles["corriger"]) == 1


def test_corriger_reserve_aux_enseignants(client, eleve):
    assert client.post("/soumissions/1/corriger-auto", headers=eleve).status_code == 403


# ─── Détection ─────────────────────────────────────────────────────────────────
def test_analyser_ia_sans_modele_pour_la_langue(client, prof, monkeypatch):
    monkeypatch.setattr(services, "detecter_ia", lambda textes: {
        "statut": "non_analysé", "langue": "fr",
        "raison": "Aucun modèle de détection pour la langue « fr »."})
    r = client.post("/soumissions/1/analyser-ia", headers=prof)
    assert r.status_code == 200
    assert r.json()["statut_analyse"] == "non_analysé"
    assert r.json()["score_ia_global"] is None
    with sqlite3.connect(main.DB) as con:
        statut, score, msg = con.execute(
            "SELECT statut_analyse, score_ia_global, message FROM RapportDetectionIA "
            "WHERE soumission_id=1").fetchone()
    assert (statut, score) == ("non_analysé", None) and "fr" in msg


# ─── services.py (adaptateur vers projet-ia) ──────────────────────────────────
def _faux_projet_ia(monkeypatch, langue, predire):
    det = types.ModuleType("projet_ia.detection")
    det.predire_textes = predire
    lang = types.ModuleType("projet_ia.detection.langue")
    lang.detecter_langue = lambda texte: langue
    monkeypatch.setitem(sys.modules, "projet_ia.detection", det)
    monkeypatch.setitem(sys.modules, "projet_ia.detection.langue", lang)


def test_detecter_ia_agrege_les_reponses(monkeypatch):
    def predire(textes, langue):
        assert langue == "en"
        return [
            {"probabilite_ia": 0.9, "seuil": 0.798, "n_phrases": 3, "phrases": [
                {"texte": "A.", "probabilite_ia": 0.95}, {"texte": "B.", "probabilite_ia": 0.5}]},
            {"probabilite_ia": 0.1, "seuil": 0.798, "n_phrases": 1,
             "avertissement": "Texte court", "phrases": [{"texte": "C.", "probabilite_ia": 0.1}]},
        ]
    _faux_projet_ia(monkeypatch, "en", predire)
    r = services.detecter_ia(["texte 1", "  ", "texte 2"])
    assert r["statut"] == "analysé"
    assert r["score"] == pytest.approx((0.9 * 3 + 0.1 * 1) / 4)
    assert r["seuil"] == 0.798
    assert r["phrases"] == [{"texte": "A.", "probabilite_ia": 0.95}]
    assert r["avertissements"] == ["Texte court"]


def test_detecter_ia_sans_modele_ou_langue_inconnue(monkeypatch):
    def sans_modele(textes, langue):
        raise FileNotFoundError("pas de modèle fr")
    _faux_projet_ia(monkeypatch, "fr", sans_modele)
    r = services.detecter_ia(["Une réponse en français."])
    assert r["statut"] == "non_analysé" and r["langue"] == "fr"

    _faux_projet_ia(monkeypatch, None, sans_modele)
    assert services.detecter_ia(["ok"])["statut"] == "non_analysé"
    assert services.detecter_ia(["", " "])["statut"] == "non_analysé"


def test_detecter_ia_erreur_technique(monkeypatch):
    def panne(textes, langue):
        raise RuntimeError("modèle corrompu")
    _faux_projet_ia(monkeypatch, "en", panne)
    r = services.detecter_ia(["some text"])
    assert r["statut"] == "erreur" and "modèle corrompu" in r["raison"]


def test_corriger_reponse_vide_sans_charger_de_modele():
    assert services.corriger_reponse("Q ?", "   ", "R")["score"] == 0.0
