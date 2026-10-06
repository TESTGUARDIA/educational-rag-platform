"""
verification/numerique.py
=========================
Pipeline numérique : extraction de nombres / expressions, comparaison exacte,
approchée ou symbolique (sympy).
"""

import math
import re

import sympy


def extraire_nombres(texte: str) -> list[float]:
    """
    Extrait toutes les valeurs numériques d'un texte.
    Gère les formats français (virgule) et anglais (point),
    les négatifs, les expressions e², sqrt(), ln().
    """
    texte_norm = texte.replace(",", ".").replace("×", "*").replace("÷", "/")
    resultats  = []

    # e², e³, e^n, e**n
    for m in re.finditer(r"e\*{1,2}(-?[\d.]+)|e\^(-?[\d.]+)|\be²|\be³", texte_norm):
        s = m.group(0)
        if s in ("e²", "e**2", "e^2"):   resultats.append(round(math.e ** 2, 6))
        elif s in ("e³", "e**3", "e^3"): resultats.append(round(math.e ** 3, 6))
        else:
            exp_val = re.search(r"(-?[\d.]+)", s)
            if exp_val:
                resultats.append(round(math.e ** float(exp_val.group(1)), 6))

    # sqrt(x), √x
    for m in re.finditer(r"sqrt\((-?[\d.]+)\)|√(-?[\d.]+)", texte_norm):
        val = m.group(1) or m.group(2)
        if val:
            resultats.append(round(math.sqrt(float(val)), 6))

    # Nombres bruts (entiers et décimaux, positifs et négatifs)
    resultats += [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", texte_norm)]
    return resultats


def extraire_expression_sympy(texte: str) -> str:
    """
    Extrait et normalise l'expression mathématique principale d'un texte pour sympy.
    Ex: "y = (x+1)²" → "(x+1)**2"
    """
    remplacements = [
        ("²", "**2"), ("³", "**3"),
        ("√", "sqrt"), ("×", "*"), ("÷", "/"),
        ("π", "pi"), ("∞", "oo"),
        ("ln(", "log("), ("e²", "exp(2)"), ("e³", "exp(3)"),
        ("⟹", ""), ("⇒", ""), ("donc", ""), ("alors", ""),
    ]
    expr = texte
    for a, b in remplacements:
        expr = expr.replace(a, b)

    # Prend la partie après le dernier "=" si présent
    if "=" in expr:
        for partie in reversed(expr.split("=")):
            partie = partie.strip()
            if partie:
                return partie
    return expr.strip()


def est_reponse_purement_mathematique(reponse: str, correction: str) -> bool:
    """
    Retourne True si la correction est purement mathématique :
    formules explicites (sqrt, log, e², π…) ou phrase très courte centrée sur un nombre.
    """
    operateurs_formels = [
        "sqrt", "racine", "log", "ln", "exp",
        "e²", "e³", "√", "π", "∑", "∫",
        "sin(", "cos(", "tan(", "f(x)",
    ]
    correction_lower = correction.lower()
    if any(op in correction_lower for op in operateurs_formels):
        return True
    if "équation" in correction_lower and "=" in correction:
        return True
    tokens  = correction.split()
    nombres = re.findall(r"-?\d+(?:[.,]\d+)?", correction)
    if len(tokens) <= 5 and len(nombres) >= 1:
        return True
    return False


def est_reponse_hybride(correction: str) -> bool:
    """
    Retourne True si la correction contient des nombres dans un contexte textuel long.
    Ex: "L'océan Pacifique fait 165 millions de km²" → hybride.
    """
    nombres = re.findall(r"-?\d+(?:[.,]\d+)?", correction)
    tokens  = correction.split()
    return len(nombres) > 0 and len(tokens) > 5


def comparer_numerique(reponse: str, correction: str) -> dict:
    """
    Compare deux réponses numériques ou algébriques.

    Stratégie en cascade :
      1. Comparaison exacte/approchée des nombres extraits
      2. Comparaison symbolique via sympy (expressions algébriques)
      3. Fallback si rien n'a pu être extrait

    Returns:
        dict avec mode, score (0-1), correct (bool), et détail
    """
    nb_rep = extraire_nombres(reponse)
    nb_cor = extraire_nombres(correction)

    # Cas 1 : comparaison numérique directe
    if nb_rep and nb_cor:
        tolerance = 1e-4
        corrects = sum(
            any(
                abs(nr - nc) <= tolerance
                or (abs(nc) > 1e-9 and abs(nr - nc) / abs(nc) < 0.01)
                for nr in nb_rep
            )
            for nc in nb_cor
        )
        score = corrects / len(nb_cor)
        return {
            "mode":          "numérique",
            "score":         round(score, 4),
            "correct":       score >= 0.99,
            "nb_reponse":    nb_rep,
            "nb_correction": nb_cor,
            "detail":        {"nb_reponse": nb_rep, "nb_correction": nb_cor},
        }

    # Cas 2 : comparaison symbolique via sympy
    try:
        expr_rep = extraire_expression_sympy(reponse)
        expr_cor = extraire_expression_sympy(correction)
        sym_rep  = sympy.sympify(expr_rep, locals={"e": sympy.E, "pi": sympy.pi})
        sym_cor  = sympy.sympify(expr_cor, locals={"e": sympy.E, "pi": sympy.pi})
        equivalent = sympy.simplify(sym_rep - sym_cor) == 0
        return {
            "mode":    "symbolique",
            "score":   1.0 if equivalent else 0.0,
            "correct": equivalent,
            "detail":  {"expr_reponse": str(sym_rep), "expr_correction": str(sym_cor)},
        }
    except Exception:
        pass

    # Cas 3 : fallback
    return {
        "mode":    "numérique_fallback",
        "score":   0.0,
        "correct": False,
        "detail":  {"raison": "Impossible d'extraire des valeurs numériques comparables"},
    }
