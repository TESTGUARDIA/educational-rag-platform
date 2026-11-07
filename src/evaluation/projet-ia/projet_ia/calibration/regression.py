"""
calibration/regression.py
=========================
Régression logistique implémentée à la main (descente de gradient).
"""

import random

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)  # evite l'overflow de exp()
    return 1.0 / (1.0 + np.exp(-z))


def cout_log_loss(y: np.ndarray, y_pred: np.ndarray, w: np.ndarray, l2: float) -> float:
    """Log-loss (entropie croisee binaire) + regularisation L2 sur les poids (hors biais)."""
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    perte  = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
    reg    = (l2 / (2 * len(y))) * np.sum(w[1:] ** 2)  # w[0] = biais, non regularise
    return perte + reg


def entrainer_descente_gradient(X: np.ndarray, y: np.ndarray,
                                 taux_apprentissage: float = 0.1,
                                 epochs: int = 5000,
                                 l2: float = 0.01,
                                 taille_batch: int | None = None,
                                 seed: int = 42,
                                 verbose: bool = True) -> dict:
    """
    Entraine une regression logistique par descente de gradient (batch ou
    mini-batch selon taille_batch).

    w[0] est le biais (le code ajoute une colonne de 1 a X).
    A chaque epoch :
        z       = X . w
        y_pred  = sigmoid(z)
        gradient = X^T . (y_pred - y) / n + L2 * w  (biais non regularise)
        w       = w - taux_apprentissage * gradient

    Retourne un dict avec les poids finaux et l'historique du cout
    (utile pour verifier la convergence / tracer une courbe).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    n, d = X.shape
    X_biais = np.hstack([np.ones((n, 1)), X])  # colonne de biais
    w = np.zeros(d + 1)

    historique_cout = []
    taille_batch = taille_batch or n  # par defaut : batch gradient descent (tout le dataset)

    for epoch in range(epochs):
        indices = rng.permutation(n)
        X_shuf  = X_biais[indices]
        y_shuf  = y[indices]

        for debut in range(0, n, taille_batch):
            X_b = X_shuf[debut:debut + taille_batch]
            y_b = y_shuf[debut:debut + taille_batch]

            z      = X_b @ w
            y_pred = sigmoid(z)
            erreur = y_pred - y_b

            gradient = (X_b.T @ erreur) / len(y_b)
            gradient[1:] += (l2 / len(y_b)) * w[1:]  # L2 sur tout sauf le biais

            w -= taux_apprentissage * gradient

        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            y_pred_full = sigmoid(X_biais @ w)
            cout = cout_log_loss(y, y_pred_full, w, l2)
            historique_cout.append({"epoch": epoch, "cout": round(float(cout), 6)})
            if verbose:
                print(f"  Epoch {epoch:>5}/{epochs}  |  cout = {cout:.6f}")

    return {"w": w, "historique_cout": historique_cout}
