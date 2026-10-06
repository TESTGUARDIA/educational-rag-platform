"""
Tests de bout en bout avec les VRAIS modèles (spaCy, CamemBERT, DeBERTa).
Lents et nécessitent les téléchargements : ignorés automatiquement s'ils manquent.
Lancer explicitement :  pytest -m modeles
"""

import pytest

pytestmark = pytest.mark.modeles

pytest.importorskip("spacy")
pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_evaluation_numerique_de_bout_en_bout():
    from projet_ia.verification import evaluer_reponse

    r = evaluer_reponse("Combien font 15 × 8 ?", "15 × 8 = 120", "Le résultat est 120.")
    assert r["verdict"] == "Correct"


def test_mauvaise_entite_detectee():
    from projet_ia.verification import evaluer_reponse

    r = evaluer_reponse("Qui a écrit Hamlet ?",
                        "Hamlet a été écrit par Victor Hugo.",
                        "Hamlet a été écrit par William Shakespeare.")
    assert r["verdict"] == "Incorrect"
