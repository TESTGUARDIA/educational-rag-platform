"""Pipeline numérique de la vérification (n'a besoin ni de spaCy ni de torch)."""

from projet_ia.verification.numerique import (
    comparer_numerique, est_reponse_hybride, est_reponse_purement_mathematique,
    extraire_nombres)


def test_extraire_nombres_formats_francais():
    assert 3.5 in extraire_nombres("Le résultat est 3,5")
    assert -2.0 in extraire_nombres("x = -2")


def test_nombre_correct_dans_une_phrase():
    r = comparer_numerique("15 × 8 = 120", "Le résultat est 120.")
    assert r["correct"] and r["score"] == 1.0


def test_nombre_incorrect():
    r = comparer_numerique("150", "120")
    assert not r["correct"]


def test_tolerance_relative_de_1_pourcent():
    assert comparer_numerique("100.5", "100")["correct"]
    assert not comparer_numerique("103", "100")["correct"]


def test_detection_du_type_de_reponse():
    assert est_reponse_purement_mathematique("x", "sqrt(16) = 4")
    assert not est_reponse_purement_mathematique("Paris", "Paris est la capitale de la France")
    assert est_reponse_hybride("L'océan Pacifique fait 165 millions de km²")
    assert not est_reponse_hybride("Paris")
