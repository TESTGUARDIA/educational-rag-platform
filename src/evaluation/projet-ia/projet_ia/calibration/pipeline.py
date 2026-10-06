"""
calibration/pipeline.py
=======================
Pipeline complet de calibration : dataset → validation croisée →
entraînement final → poids sur features brutes → poids_calibres.json.
"""

import json

import numpy as np

from .. import config
from ..common.fichiers import creer_dossier_parent
from .dataset import FEATURES_NUMERIQUES, charger_dataset, normaliser_standard
from .evaluation import evaluer_modele, validation_croisee_k_fold
from .regression import entrainer_descente_gradient


def calibrer(chemin_dataset: str = str(config.CHEMIN_META_DATASET),
             chemin_sortie: str = str(config.CHEMIN_POIDS_CALIBRES),
             taux_apprentissage: float = 0.1,
             epochs: int = 5000,
             l2: float = 0.01,
             k_fold: int = 5,
             seuil_decision: float = 0.5) -> dict:

    print(f"\n{'='*65}")
    print("  CALIBRATION DES POIDS — DESCENTE DE GRADIENT")
    print(f"{'='*65}")

    print(f"\nChargement du dataset : {chemin_dataset}")
    X, y, noms_colonnes, categories = charger_dataset(chemin_dataset)
    print(f"  {X.shape[0]} exemples, {X.shape[1]} features")
    print(f"  Features      : {FEATURES_NUMERIQUES}")
    if categories:
        print(f"  Categories one-hot ({len(categories)}) : {categories}")
    print(f"  Repartition label : {int(y.sum())} positifs / {int(len(y) - y.sum())} negatifs")

    # ── Validation croisee (estimation de la robustesse) ─────────────────────
    print(f"\nValidation croisee {k_fold}-fold...")
    cv = validation_croisee_k_fold(
        X, y, k=k_fold, taux_apprentissage=taux_apprentissage,
        epochs=max(500, epochs // 2), l2=l2,
    )
    print(f"  F1  = {cv['f1_moyen']:.4f} +/- {cv['f1_ecart_type']:.4f}")
    print(f"  Acc = {cv['acc_moyen']:.4f} +/- {cv['acc_ecart_type']:.4f}")

    # ── Entrainement final sur tout le dataset ────────────────────────────────
    print(f"\nEntrainement final sur l'ensemble du dataset ({epochs} epochs)...")
    X_norm, moyennes, ecarts = normaliser_standard(X)
    resultat = entrainer_descente_gradient(
        X_norm, y, taux_apprentissage=taux_apprentissage,
        epochs=epochs, l2=l2, verbose=True,
    )
    w = resultat["w"]

    metriques_train = evaluer_modele(X_norm, y, w, seuil=seuil_decision)
    print(f"\nMetriques sur le dataset complet (apres entrainement) :")
    print(f"  Accuracy  = {metriques_train['accuracy']:.4f}")
    print(f"  Precision = {metriques_train['precision']:.4f}")
    print(f"  Rappel    = {metriques_train['rappel']:.4f}")
    print(f"  F1        = {metriques_train['f1']:.4f}")

    # ── Reconversion des poids standardises -> poids "bruts" ─────────────────
    # Le modele est entraine sur X_norm = (X - moyenne) / ecart_type.
    # Pour pouvoir appliquer les poids directement sur des features brutes
    # (comme le fait verification_reponse_finale.py), on "deplie" la
    # standardisation dans les poids et le biais :
    #   z = b + sum( w_i * (x_i - mu_i) / sigma_i )
    #     = (b - sum(w_i * mu_i / sigma_i)) + sum( (w_i / sigma_i) * x_i )
    biais_std = w[0]
    poids_std = w[1:]

    poids_bruts = poids_std / ecarts
    biais_brut  = biais_std - float(np.sum(poids_std * moyennes / ecarts))

    poids_calibres = {nom: round(float(p), 4) for nom, p in zip(noms_colonnes, poids_bruts)}
    poids_calibres["biais"] = round(float(biais_brut), 4)

    # S'assurer que toutes les categories "standard" existent dans le dict
    # de sortie (a 0.0 si absentes du dataset), pour rester compatible avec
    # verification_reponse_finale.py qui s'attend a q_fermee/q_semi_ouverte/q_ouverte.
    for cat_standard in ["q_fermee", "q_semi_ouverte", "q_ouverte"]:
        poids_calibres.setdefault(cat_standard, 0.0)

    print(f"\nPoids calibres (sur features brutes, pretes a l'emploi) :")
    for nom, val in poids_calibres.items():
        print(f"  {nom:<24} : {val:+.4f}")

    sortie = {
        "poids":               poids_calibres,
        "features_numeriques": FEATURES_NUMERIQUES,
        "categories_one_hot":  categories,
        "metriques_entrainement": metriques_train,
        "validation_croisee":     cv,
        "hyperparametres": {
            "taux_apprentissage": taux_apprentissage,
            "epochs":             epochs,
            "l2":                 l2,
            "seuil_decision":     seuil_decision,
            "k_fold":             k_fold,
        },
        "n_exemples_entrainement": int(X.shape[0]),
    }

    creer_dossier_parent(chemin_sortie)
    with open(chemin_sortie, "w", encoding="utf-8") as f:
        json.dump(sortie, f, indent=2, ensure_ascii=False)

    print(f"\nPoids sauvegardes : {chemin_sortie}")
    print(f"{'='*65}")
    return sortie
