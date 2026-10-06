"""
detection/fusion.py
===================
Fusion des composants : somme pondérée de leurs log-odds, poids appris par
DESCENTE DE GRADIENT (la régression logistique de calibration/regression.py).

Entrées de la fusion, pour chaque paragraphe (toutes en log-odds) :
    paragraphe     sortie du modèle-paragraphe
    interphrases   sortie du modèle de comparaison entre phrases
    phrases_moy    moyenne des log-odds des phrases
    phrases_max    phrase la plus « IA »
    phrases_min    phrase la plus « humaine »

Chaque entrée z compte DEUX fois, avec deux poids distincts :
    z+ = max(z, 0)   évidence « IA »      → poids_ia
    z- = min(z, 0)   évidence « humain »  → poids_humain
        score = biais + Σ poids_ia·z+ + Σ poids_humain·z-        p(IA) = sigmoid(score)
C'est ce qui permet qu'un composant « croie » davantage ses alertes IA que ses
verdicts humains (ou l'inverse), et le biais décale le seuil global.

Pourquoi des log-odds et pas des probabilités : une probabilité de 0.99 et une de
0.9999 ne diffèrent presque pas, alors que leur confiance est très différente.
Sommer des log-odds accumule l'évidence sans cet écrasement.
Le problème reste une régression logistique : convexe, donc la descente de gradient
atteint l'optimum, sans minimum local.
"""

import numpy as np

from ..calibration.regression import entrainer_descente_gradient, sigmoid

EPS = 1e-6
NOMS_ENTREES = ["paragraphe", "interphrases", "phrases_moy", "phrases_max", "phrases_min"]


def logit(p) -> np.ndarray:
    """Log-odds d'une probabilité (bornée pour éviter l'infini)."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def agreger_phrases(logits_phrases, idx_texte, n_textes: int) -> dict:
    """
    Résume les log-odds des phrases de chaque paragraphe : moyenne, maximum, minimum.
    (Un paragraphe sans phrase reçoit 0 = neutre.)
    """
    z = np.asarray(logits_phrases, dtype=float)
    idx = np.asarray(idx_texte)
    n = np.bincount(idx, minlength=n_textes)
    somme = np.bincount(idx, weights=z, minlength=n_textes)
    moy = np.divide(somme, n, out=np.zeros(n_textes), where=n > 0)
    mx, mn = np.full(n_textes, -np.inf), np.full(n_textes, np.inf)
    np.maximum.at(mx, idx, z)
    np.minimum.at(mn, idx, z)
    vide = n == 0
    mx[vide] = mn[vide] = 0.0
    return {"phrases_moy": moy, "phrases_max": mx, "phrases_min": mn}


def assembler_entrees(logits: dict) -> np.ndarray:
    """dict nom -> log-odds (n,)  →  tableau (n, 5), colonnes dans l'ordre NOMS_ENTREES."""
    return np.column_stack([np.asarray(logits[nom], dtype=float) for nom in NOMS_ENTREES])


def _developper(Z: np.ndarray) -> np.ndarray:
    """(n, 5) → (n, 10) : [parties positives (IA) | parties négatives (humain)]."""
    return np.hstack([np.maximum(Z, 0.0), np.minimum(Z, 0.0)])


def entrainer_fusion(Z, y, taux_apprentissage: float = 0.1, epochs: int = 3000,
                     l2: float = 1.0, verbose: bool = False) -> dict:
    """
    Apprend les poids de fusion par descente de gradient.

    Args:
        Z : (n, 5) log-odds des composants (voir assembler_entrees)
        y : labels 0/1
    Returns:
        dict : poids_ia, poids_humain (par entrée), biais, historique_cout
    """
    X = _developper(np.asarray(Z, dtype=float))
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
    res = entrainer_descente_gradient((X - mu) / sd, np.asarray(y, dtype=float),
                                      taux_apprentissage=taux_apprentissage, epochs=epochs,
                                      l2=l2, verbose=verbose)
    w = res["w"]
    poids = w[1:] / sd                                  # retour à l'échelle des log-odds bruts
    biais = float(w[0] - np.sum(w[1:] * mu / sd))
    k = len(NOMS_ENTREES)
    return {"noms": list(NOMS_ENTREES),
            "poids_ia": dict(zip(NOMS_ENTREES, map(float, poids[:k]))),
            "poids_humain": dict(zip(NOMS_ENTREES, map(float, poids[k:]))),
            "biais": biais, "historique_cout": res["historique_cout"]}


def predire_fusion(fusion: dict, Z) -> np.ndarray:
    """Probabilité « IA » finale pour chaque ligne de Z."""
    X = _developper(np.asarray(Z, dtype=float))
    w = np.array([fusion["poids_ia"][n] for n in fusion["noms"]] +
                 [fusion["poids_humain"][n] for n in fusion["noms"]])
    return sigmoid(X @ w + fusion["biais"])
