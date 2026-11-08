"""Vérificateur numérique des corrigés (src/api/problemes/verification.py)."""

import pytest

from src.api.problemes.verification import (ExpressionRefusee, comparer_valeurs, lire_expression,
                                            verifier_probleme)

WALLIS = {"nom": "W", "parametre": "n", "variable": "x",
          "integrande": "sin(x)^n", "borne_inf": "0", "borne_sup": "pi/2"}


def _probleme(*verifs):
    return {"definitions": [WALLIS], "verifications": list(verifs)}


@pytest.mark.parametrize("verif", [
    {"type": "egalite", "gauche": "W(0)", "droite": "pi/2"},
    {"type": "egalite", "gauche": "W(1)", "droite": "1"},
    {"type": "egalite", "gauche": "W(n+2)", "droite": "(n+1)/(n+2)*W(n)", "n_min": 0, "n_max": 12},
    {"type": "egalite", "gauche": "(n+1)*W(n+1)*W(n)", "droite": "pi/2", "n_min": 0, "n_max": 10},
    {"type": "egalite", "gauche": "W(2n)", "droite": "factorial(2n)/(2^(2n)*factorial(n)^2)*pi/2",
     "n_min": 0, "n_max": 8},
    {"type": "inegalite", "gauche": "W(n+1)", "droite": "W(n)", "operateur": "<=", "n_min": 0, "n_max": 15},
    {"type": "limite", "gauche": "sqrt(n)*W(n)", "droite": "sqrt(pi/2)"},
])
def test_proprietes_vraies_de_wallis(verif):
    r = verifier_probleme(_probleme(verif))
    assert r["ok"] == 1, r["resultats"]


@pytest.mark.parametrize("verif", [
    {"type": "egalite", "gauche": "W(n+2)", "droite": "n/(n+2)*W(n)", "n_min": 0, "n_max": 6},
    {"type": "egalite", "gauche": "W(1)", "droite": "pi/4"},
    {"type": "inegalite", "gauche": "W(n)", "droite": "W(n+1)", "operateur": "<=", "n_min": 0, "n_max": 5},
    {"type": "limite", "gauche": "n*W(n)", "droite": "sqrt(pi/2)"},
])
def test_affirmations_fausses_detectees(verif):
    r = verifier_probleme(_probleme(verif))
    assert r["faux"] == 1, r["resultats"]


def test_suite_d_integrales_du_cours():
    # I_n = intégrale de x (ln x)^n entre 1 et e : I_{n+1} = e^2/2 - (n+1)/2 I_n
    pb = {"definitions": [{"nom": "I", "integrande": "x*ln(x)^n", "borne_inf": "1", "borne_sup": "e"}],
          "verifications": [
              {"type": "egalite", "gauche": "I(0)", "droite": "(e^2-1)/2"},
              {"type": "egalite", "gauche": "I(n+1)", "droite": "e^2/2-(n+1)/2*I(n)", "n_max": 8}]}
    assert verifier_probleme(pb)["ok"] == 2


@pytest.mark.parametrize("texte", [
    "__import__('os').system('dir')",
    "sin.__class__",
    "pi.evalf",
    "open(1)",
    "exec(1)",
    "lambda: 1",
    "x[0]",
    "x" * 500,
])
def test_expressions_dangereuses_refusees(texte):
    with pytest.raises(ExpressionRefusee):
        lire_expression(texte)


def test_verification_illisible_ne_plante_pas():
    r = verifier_probleme(_probleme({"type": "egalite", "gauche": "W(n", "droite": "1"},
                                    {"type": "inconnu", "gauche": "1", "droite": "1"},
                                    {"type": "egalite", "gauche": "W(-1)", "droite": "1"}))
    assert r["erreur"] == 3


def test_definition_invalide_signalee():
    r = verifier_probleme({"definitions": [{"nom": "os", "integrande": "x", "borne_inf": "0",
                                            "borne_sup": "1"}], "verifications": []})
    assert r["erreurs_definitions"]


def test_comparer_valeurs():
    assert comparer_valeurs("pi^2/4 - 2", "(pi/2)^2-2") is True
    assert comparer_valeurs("e^2+1", "e^2-1") is False
    assert comparer_valeurs("la suite diverge", "1") is None


# ─── Réparation du LaTeX dans le JSON du modèle (ollama.py) ───────────────────
import json  # noqa: E402

from src.api.problemes.ollama import reparer_latex  # noqa: E402


@pytest.mark.parametrize("brut, attendu", [
    # Sortie réelle du modèle : \\int bien doublé (\i serait un JSON invalide, interdit par
    # le schéma) mais \text non doublé, donc lu comme tabulation + « ext »
    (r'{"e": "$\\int_a^b f(x)\text{d}x$"}', r"$\int_a^b f(x)\text{d}x$"),
    (r'{"e": "$\frac{1}{2}$"}', r"$\frac{1}{2}$"),                           # \f de \frac
    (r'{"e": "$a \neq b$, $\beta$, $\rho$"}', r"$a \neq b$, $\beta$, $\rho$"),
    (r'{"e": "$\\frac{1}{2}$ déjà correct"}', r"$\frac{1}{2}$ déjà correct"),
    # Sorties réelles de llama3.2:3b : lettre forcée par la grammaire devant \int
    (r'{"e": "$m=\\frac{1}{b-a}\tint_a^bf(x)\text{d}x$"}', r"$m=\frac{1}{b-a}\int_a^bf(x)\text{d}x$"),
    (r'{"e": "$I_n=\tint_1^e x\tsqrt{x}$"}', r"$I_n=\int_1^e x\sqrt{x}$"),
    (r'{"e": "ligne 1\nligne 2"}', "ligne 1\nligne 2"),                      # vrai saut de ligne
    (r'{"e": "Soit f.\nle cours dit"}', "Soit f.\nle cours dit"),            # « le » ≠ \le
    (r'{"e": "a\tb"}', "a\tb"),                                               # vraie tabulation
])
def test_reparer_latex(brut, attendu):
    assert json.loads(reparer_latex(brut))["e"] == attendu
