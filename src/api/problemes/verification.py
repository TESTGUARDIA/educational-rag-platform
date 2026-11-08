"""
Vérification numérique des affirmations d'un corrigé (SymPy + mpmath).

Le modèle de génération ne fournit jamais de code : il déclare
  - des familles d'intégrales (ex. W(n) = intégrale de sin(x)^n entre 0 et pi/2),
  - des vérifications typées : égalité, inégalité ou limite, en fonction de n.

Chaque expression est filtrée par une liste blanche (fonctions et noms autorisés)
avant d'être lue par SymPy, puis évaluée numériquement pour plusieurs valeurs de n.
Une affirmation fausse du corrigé (relation de récurrence, valeur, limite…) est
ainsi détectée sans faire confiance au modèle.
"""

import re
from dataclasses import dataclass, field

import mpmath
import sympy
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

mpmath.mp.dps = 30

# Noms autorisés dans les expressions (en plus des familles déclarées et des variables)
FONCTIONS = {
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan, "exp": sympy.exp,
    "ln": sympy.log, "log": sympy.log, "sqrt": sympy.sqrt, "abs": sympy.Abs,
    "factorial": sympy.factorial, "binomial": sympy.binomial,
    "atan": sympy.atan, "asin": sympy.asin, "acos": sympy.acos,
    "sinh": sympy.sinh, "cosh": sympy.cosh,
}
CONSTANTES = {"pi": sympy.pi, "e": sympy.E, "E": sympy.E, "oo": sympy.oo}
VARIABLES = ("n", "p", "k", "x", "t")

_CARACTERES = re.compile(r"^[0-9A-Za-z_+\-*/^().,\s]*$")
_IDENTIFIANT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_POINT_ATTRIBUT = re.compile(r"(?<![0-9])\.|\.(?![0-9])")   # « . » seulement dans 3.14
_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)

TOLERANCE = 1e-8          # écart relatif toléré pour une égalité
TOLERANCE_LIMITE = 2e-2   # écart relatif toléré à n grand pour une limite


class ExpressionRefusee(ValueError):
    """Expression hors de la liste blanche, ou illisible."""


@dataclass
class Famille:
    """Famille d'intégrales : nom(parametre) = intégrale de integrande pour variable de a à b."""
    nom: str
    parametre: str
    variable: str
    integrande: sympy.Expr
    borne_inf: sympy.Expr
    borne_sup: sympy.Expr
    _cache: dict = field(default_factory=dict, repr=False)

    def valeur(self, k: int) -> mpmath.mpf:
        if k not in self._cache:
            f = self.integrande.subs(sympy.Symbol(self.parametre), k)
            x = sympy.Symbol(self.variable)
            g = sympy.lambdify(x, f, modules="mpmath")
            a = mpmath.mpf(sympy.N(self.borne_inf, 30))
            b = mpmath.mpf(sympy.N(self.borne_sup, 30))
            # Découpage de l'intervalle : intégrandes très piquées pour k grand (sin^k)
            points = [a + (b - a) * i / 8 for i in range(9)]
            self._cache[k] = mpmath.quad(g, points)
        return self._cache[k]


def lire_expression(texte: str, familles: dict[str, Famille] | None = None) -> sympy.Expr:
    """Lit une expression mathématique écrite en syntaxe Python/SymPy, après filtrage."""
    familles = familles or {}
    texte = (texte or "").strip()
    if not texte or len(texte) > 400:
        raise ExpressionRefusee("expression vide ou trop longue")
    if not _CARACTERES.match(texte) or "__" in texte or _POINT_ATTRIBUT.search(texte):
        raise ExpressionRefusee(f"caractères non autorisés : {texte!r}")
    autorises = set(FONCTIONS) | set(CONSTANTES) | set(VARIABLES) | set(familles)
    inconnus = {m for m in _IDENTIFIANT.findall(texte) if m not in autorises}
    if inconnus:
        raise ExpressionRefusee(f"noms non autorisés : {sorted(inconnus)}")
    locaux = {**FONCTIONS, **CONSTANTES,
              **{v: sympy.Symbol(v) for v in VARIABLES},
              **{nom: sympy.Function(nom) for nom in familles}}
    try:
        return parse_expr(texte, local_dict=locaux, global_dict={"__builtins__": {}, **_GLOBAUX},
                          transformations=_TRANSFORMATIONS, evaluate=True)
    except Exception as e:  # erreur de syntaxe, etc.
        raise ExpressionRefusee(f"expression illisible {texte!r} : {e}") from e


# Noms dont parse_expr a besoin pour construire les expressions (pas de builtins)
_GLOBAUX = {name: getattr(sympy, name) for name in
            ("Integer", "Float", "Rational", "Symbol", "Function")}


def construire_famille(d: dict) -> Famille:
    nom = str(d["nom"]).strip()
    if not re.fullmatch(r"[A-Z][A-Za-z0-9]{0,3}", nom):
        raise ExpressionRefusee(f"nom de famille invalide : {nom!r} (ex. W, I, J)")
    return Famille(nom=nom, parametre=d.get("parametre", "n"), variable=d.get("variable", "x"),
                   integrande=lire_expression(d["integrande"]),
                   borne_inf=lire_expression(str(d["borne_inf"])),
                   borne_sup=lire_expression(str(d["borne_sup"])))


def evaluer(expr: sympy.Expr, familles: dict[str, Famille], valeurs: dict[str, int]) -> mpmath.mpf:
    """Valeur numérique de expr pour des valeurs entières des paramètres (n, p, k)."""
    e = expr.subs({sympy.Symbol(k): v for k, v in valeurs.items()})
    # Remplace chaque W(argument) par la valeur numérique de l'intégrale
    for appel in sorted(e.atoms(sympy.core.function.AppliedUndef), key=str):
        nom = appel.func.__name__
        if nom not in familles:
            raise ExpressionRefusee(f"famille inconnue : {nom}")
        arg = sympy.N(appel.args[0])
        if not arg.is_number or abs(arg - round(arg)) > 1e-9 or round(arg) < 0:
            raise ExpressionRefusee(f"{appel} : l'indice doit être un entier positif")
        e = e.subs(appel, sympy.Float(familles[nom].valeur(int(round(arg))), 30))
    v = sympy.N(e, 30)
    if not v.is_real:
        raise ExpressionRefusee(f"valeur non réelle : {v}")
    return mpmath.mpf(v)


def _proches(a, b, tol=TOLERANCE) -> bool:
    return abs(a - b) <= tol * max(1, abs(a), abs(b))


def verifier(verif: dict, familles: dict[str, Famille]) -> dict:
    """
    Exécute une vérification. Format attendu (produit par le modèle) :
      {"type": "egalite"|"inegalite"|"limite", "gauche": "...", "droite": "...",
       "operateur": "<="|"<"|">="|">" (inégalité), "n_min": 0, "n_max": 10}
    Pour une limite : "gauche" est l'expression en n, "droite" la limite (ou "oo").
    """
    t = verif.get("type")
    res = {"type": t, "question": verif.get("question"), "description": verif.get("description", ""),
           "gauche": verif.get("gauche"), "droite": verif.get("droite")}
    try:
        g = lire_expression(verif["gauche"], familles)
        d = lire_expression(verif["droite"], familles)
        n_min = max(0, int(verif.get("n_min", 0)))
        n_max = min(int(verif.get("n_max", 10)), n_min + 40)
        utilise_n = sympy.Symbol("n") in (g.free_symbols | d.free_symbols) or any(
            sympy.Symbol("n") in a.free_symbols for a in (g.atoms(sympy.core.function.AppliedUndef)
                                                        | d.atoms(sympy.core.function.AppliedUndef)))

        if t == "egalite":
            echecs = []
            for n in (range(n_min, n_max + 1) if utilise_n else [0]):
                vg, vd = evaluer(g, familles, {"n": n}), evaluer(d, familles, {"n": n})
                if not _proches(vg, vd):
                    echecs.append(f"n={n} : {mpmath.nstr(vg, 10)} ≠ {mpmath.nstr(vd, 10)}")
            res.update(statut="ok" if not echecs else "faux", detail=echecs[:3])

        elif t == "inegalite":
            op = verif.get("operateur", "<=")
            comparer = {"<=": lambda a, b: a <= b + TOLERANCE * max(1, abs(b)),
                        "<": lambda a, b: a < b,
                        ">=": lambda a, b: a + TOLERANCE * max(1, abs(a)) >= b,
                        ">": lambda a, b: a > b}[op]
            echecs = []
            for n in (range(n_min, n_max + 1) if utilise_n else [0]):
                vg, vd = evaluer(g, familles, {"n": n}), evaluer(d, familles, {"n": n})
                if not comparer(vg, vd):
                    echecs.append(f"n={n} : {mpmath.nstr(vg, 10)} {op} {mpmath.nstr(vd, 10)} est faux")
            res.update(statut="ok" if not echecs else "faux", detail=echecs[:3])

        elif t == "limite":
            n1, n2 = 100, 400
            v1, v2 = evaluer(g, familles, {"n": n1}), evaluer(g, familles, {"n": n2})
            if d == sympy.oo:
                ok = v2 > v1 and v2 > 1e3
                detail = [f"u({n1})={mpmath.nstr(v1, 8)}, u({n2})={mpmath.nstr(v2, 8)}"]
            else:
                lim = evaluer(d, familles, {})
                e1, e2 = abs(v1 - lim), abs(v2 - lim)
                ok = e2 <= TOLERANCE_LIMITE * max(1, abs(lim)) and e2 <= e1 + 1e-12
                detail = [f"u({n1})={mpmath.nstr(v1, 8)}, u({n2})={mpmath.nstr(v2, 8)}, "
                          f"limite annoncée={mpmath.nstr(lim, 8)}"]
            res.update(statut="ok" if ok else "faux", detail=detail)
        else:
            res.update(statut="erreur", detail=[f"type de vérification inconnu : {t!r}"])
    except ExpressionRefusee as e:
        res.update(statut="erreur", detail=[str(e)])
    except (ZeroDivisionError, OverflowError, ValueError, TypeError, KeyError) as e:
        res.update(statut="erreur", detail=[f"{type(e).__name__} : {e}"])
    return res


def verifier_probleme(probleme: dict) -> dict:
    """Construit les familles du problème puis exécute toutes ses vérifications."""
    familles, erreurs = {}, []
    for d in probleme.get("definitions", []):
        try:
            f = construire_famille(d)
            familles[f.nom] = f
        except (ExpressionRefusee, KeyError) as e:
            erreurs.append(f"définition {d.get('nom')!r} : {e}")
    resultats = [verifier(v, familles) for v in probleme.get("verifications", [])]
    compte = {s: sum(r["statut"] == s for r in resultats) for s in ("ok", "faux", "erreur")}
    return {"familles": sorted(familles), "erreurs_definitions": erreurs,
            "resultats": resultats, **compte}


def comparer_valeurs(a: str, b: str, familles: dict[str, Famille] | None = None) -> bool | None:
    """True/False si les deux réponses sont lisibles et comparables, None sinon."""
    familles = familles or {}
    try:
        ea, eb = lire_expression(a, familles), lire_expression(b, familles)
        ns = [3, 4, 7] if sympy.Symbol("n") in (ea.free_symbols | eb.free_symbols) else [0]
        return all(_proches(evaluer(ea, familles, {"n": n}), evaluer(eb, familles, {"n": n}), 1e-6)
                   for n in ns)
    except (ExpressionRefusee, ZeroDivisionError, OverflowError, ValueError, TypeError):
        return None


__all__ = ["Famille", "ExpressionRefusee", "lire_expression", "construire_famille",
           "evaluer", "verifier", "verifier_probleme", "comparer_valeurs"]
