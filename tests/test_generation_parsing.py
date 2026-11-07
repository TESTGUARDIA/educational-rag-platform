"""Analyse des réponses JSON du modèle de génération (src/api/rag.py)."""

import pytest

from src.api.rag import _parse_model_response


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
