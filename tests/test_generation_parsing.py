"""Analyse des réponses JSON du modèle de génération (src/generation/rag.py)."""

import pytest

from src.generation.rag import _parse_model_response


def test_format_attendu():
    r = _parse_model_response('{"questions": [{"question": "Q ?", "reponse": "R"}]}')
    assert r == [{"question": "Q ?", "reponse": "R"}]


def test_liste_melee_d_objets_et_de_chaines():
    # Réponse réelle de llama3.2:3b (7 nov. 2026) : une chaîne au milieu de la liste
    brut = ('{"questions": [{"question": "Quels outils ?", "reponse": "keras"}, "source"],'
            '"question": "Quel but ?", "reponse": "classer"}')
    assert _parse_model_response(brut) == [{"question": "Quels outils ?", "reponse": "keras"}]


@pytest.mark.parametrize("brut, attendu", [
    ('[{"question": "Q ?", "reponse": "R"}]', 1),          # liste directe
    ('{"question": "Q ?", "reponse": "R"}', 1),            # question seule
    ('{"items": [{"question": "Q ?"}]}', 1),               # autre clé
    ('{"questions": {"question": "Q ?"}}', 1),             # objet au lieu de liste
    ('{"questions": "aucune"}', 0),
    ('{"questions": [{"question": "  "}, {"reponse": "R"}]}', 0),
    ('{}', 0),
])
def test_formats_tolerés(brut, attendu):
    assert len(_parse_model_response(brut)) == attendu


def test_json_invalide():
    assert _parse_model_response("pas du json") is None


def test_latex_non_echappe():
    r = _parse_model_response(r'{"questions": [{"question": "Q ?", "reponse": "$\frac{1}{2}$"}]}')
    assert r[0]["reponse"] == r"$\frac{1}{2}$"


def test_nb_questions_respecte(monkeypatch):
    """Le modèle renvoie 3 questions pour une section qui n'en demande qu'une."""
    import json
    from types import SimpleNamespace

    from src.generation import rag

    pages = [SimpleNamespace(page_content="Texte du cours. " * 100, metadata={})]
    monkeypatch.setattr(rag, "PyPDFLoader", lambda path: SimpleNamespace(load=lambda: pages))
    reponse = json.dumps({"questions": [{"question": f"Q{i} ?", "reponse": "R"} for i in range(3)]})

    class FauxLLM:
        def __init__(self, **kw): pass
        def invoke(self, messages): return SimpleNamespace(content=reponse)

    monkeypatch.setattr(rag, "ChatOllama", FauxLLM)
    for nb in (1, 2, 5):
        assert len(rag.ingest_and_generate_questions(b"%PDF", "cours.pdf", nb_questions=nb)) <= nb
