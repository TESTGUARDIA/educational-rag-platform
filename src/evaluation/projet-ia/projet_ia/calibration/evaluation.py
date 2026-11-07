"""
calibration/evaluation.py
=========================
Métriques (accuracy, précision, rappel, F1) et validation croisée k-fold.
"""

import numpy as np

from .dataset import normaliser_standard
from .regression import entrainer_descente_gradient, sigmoid


def evaluer_modele(X: np.ndarray, y: np.ndarray, w: np.ndarray, seuil: float = 0.5) -> dict:
    n = X.shape[0]
    X_biais = np.hstack([np.ones((n, 1)), X])
    y_pred_proba = sigmoid(X_biais @ w)
    y_pred = (y_pred_proba >= seuil).astype(float)

    vp = np.sum((y_pred == 1) & (y == 1))
    vn = np.sum((y_pred == 0) & (y == 0))
    fp = np.sum((y_pred == 1) & (y == 0))
    fn = np.sum((y_pred == 0) & (y == 1))

    accuracy  = (vp + vn) / n
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    rappel    = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    f1        = (2 * precision * rappel / (precision + rappel)
                 if (precision + rappel) > 0 else 0.0)

    return {
        "accuracy":  round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "rappel":    round(float(rappel), 4),
        "f1":        round(float(f1), 4),
        "matrice_confusion": {
            "vrais_positifs":  int(vp), "vrais_negatifs":  int(vn),
            "faux_positifs":   int(fp), "faux_negatifs":   int(fn),
        },
    }


def validation_croisee_k_fold(X: np.ndarray, y: np.ndarray, k: int = 5,
                               taux_apprentissage: float = 0.1,
                               epochs: int = 2000, l2: float = 0.01,
                               seed: int = 42) -> dict:
    """
    Validation croisee k-fold pour estimer la robustesse du modele
    (evite de juger la performance uniquement sur les donnees d'entrainement).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    indices = rng.permutation(n)
    folds = np.array_split(indices, k)

    scores_f1 = []
    scores_acc = []

    for i in range(k):
        idx_test  = folds[i]
        idx_train = np.concatenate([folds[j] for j in range(k) if j != i])

        X_train, y_train = X[idx_train], y[idx_train]
        X_test,  y_test  = X[idx_test],  y[idx_test]

        X_train_norm, moyennes, ecarts = normaliser_standard(X_train)
        X_test_norm = (X_test - moyennes) / ecarts

        resultat = entrainer_descente_gradient(
            X_train_norm, y_train,
            taux_apprentissage=taux_apprentissage, epochs=epochs,
            l2=l2, seed=seed, verbose=False,
        )
        metriques = evaluer_modele(X_test_norm, y_test, resultat["w"])
        scores_f1.append(metriques["f1"])
        scores_acc.append(metriques["accuracy"])

    return {
        "f1_moyen":       round(float(np.mean(scores_f1)), 4),
        "f1_ecart_type":  round(float(np.std(scores_f1)), 4),
        "acc_moyen":      round(float(np.mean(scores_acc)), 4),
        "acc_ecart_type": round(float(np.std(scores_acc)), 4),
        "scores_f1_par_fold": scores_f1,
    }
