"""
Fixtures communes : backend web sur une base SQLite temporaire, sans modèles.

Les trois branchements inter-pôles (services.generer_questions, corriger_reponse,
detecter_ia) sont remplacés par des doublures : les tests vérifient le câblage
de l'API, pas la qualité des modèles (mesurée par chaque pôle).
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

_TMP = tempfile.mkdtemp(prefix="testguard-")
os.environ.setdefault("JWT_SECRET", "secret-de-test")
os.environ["DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["UPLOADS_DIR"] = os.path.join(_TMP, "uploads")
os.environ["PRECHARGER_MODELES"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from src.web.back import main, services  # noqa: E402

PDF = b"%PDF-1.4\n% faux PDF de test\n"


@pytest.fixture()
def client():
    if os.path.exists(main.DB):
        os.remove(main.DB)
    with TestClient(main.app) as c:     # déclenche le startup (init_db)
        yield c


def _token(client, email):
    r = client.post("/auth/login", data={"username": email, "password": "password"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture()
def prof(client):
    return _token(client, "marie.dupont@testguard.ai")


@pytest.fixture()
def eleve(client):
    return _token(client, "lucas.moreau@testguard.ai")


@pytest.fixture()
def faux_modeles(monkeypatch):
    """Doublures des pôles Génération et Évaluation ; enregistre les appels."""
    appels = {"generer": [], "corriger": [], "detecter": []}

    def generer(pdf, nom, nb):
        appels["generer"].append((nom, nb))
        return [{"question": f"Question {i} ?", "reponse": f"Réponse {i}"} for i in range(1, nb + 1)]

    def corriger(question, reponse, correction):
        appels["corriger"].append((question, reponse, correction))
        score = 1.0 if reponse.strip() == correction.strip() else 0.5
        return {"score": score, "correct": score >= 0.6, "verdict": "", "mode": "test"}

    def detecter(textes):
        appels["detecter"].append(list(textes))
        return {"statut": "analysé", "score": 0.7, "seuil": 0.798, "langue": "en",
                "phrases": [], "avertissements": []}

    monkeypatch.setattr(services, "generer_questions", generer)
    monkeypatch.setattr(services, "corriger_reponse", corriger)
    monkeypatch.setattr(services, "detecter_ia", detecter)
    return appels
