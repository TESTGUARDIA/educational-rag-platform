"""Découpage d'un cours en parties (src/generation/problemes/analyse.py)."""

from src.generation.problemes.analyse import decouper_parties

INTRO = "Ce chapitre présente les intégrales. " * 10


def _titres(parties):
    return [(p["numero"], p["titre"]) for p in parties]


def test_style_monka():
    cours = "# Partie 1 : Primitives\nbla\n# Partie 2 : Intégrale et aire\nbla"
    assert _titres(decouper_parties(cours)) == [(1, "Primitives"), (2, "Intégrale et aire")]


def test_titres_markdown():
    cours = f"{INTRO}\n# Riemann\ntexte\n# Primitive\ntexte\n## sous-partie\ntexte"
    assert _titres(decouper_parties(cours)) == [(0, "Introduction"), (1, "Riemann"), (2, "Primitive")]


def test_titres_numerotes_non_pris_pour_des_parties():
    # Couche texte Exo7 : sous-questions et en-têtes de page ressemblent à des titres
    cours = ("Intégrales\n1. L’intégrale de Riemann\ntexte\n1. Montrer que f est intégrable\n"
             "L’INTÉGRALE DE RIEMANN 2\n2. Calculer I\n2. Propriétés\ntexte")
    assert _titres(decouper_parties(cours)) == [(1, "Cours (morceau 1)")]


def test_vision_exo7_titres_markdown_niveau2():
    cours = "# Intégrales\n## 1. L’intégrale de Riemann\nt\n## 2. Propriétés\nt\n## 3. Primitive\nt"
    assert [t for _, t in _titres(decouper_parties(cours))] == [
        "1. L’intégrale de Riemann", "2. Propriétés", "3. Primitive"]


def test_cours_sans_structure_coupe_en_morceaux():
    cours = "\n\n".join(["paragraphe " * 100] * 30)          # ~33 000 caractères
    parties = decouper_parties(cours, taille_max=12000)
    assert len(parties) == 3 and all(len(p["texte"]) <= 12000 for p in parties)
    assert "".join(p["texte"] for p in parties).count("paragraphe") == 3000   # rien de perdu


def test_partie_trop_longue_coupee_sans_perte():
    long = "\n\n".join(["x" * 1000] * 30)
    parties = decouper_parties(f"# Partie 1 : A\n{long}\n# Partie 2 : B\ncourt", taille_max=12000)
    assert [p["numero"] for p in parties] == [1, 1, 1, 2]
    assert parties[1]["titre"] == "A (suite 2)"
    assert sum(p["texte"].count("x") for p in parties) == 30000
