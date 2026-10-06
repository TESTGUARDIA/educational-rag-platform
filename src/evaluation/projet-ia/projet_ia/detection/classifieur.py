"""
detection/classifieur.py
========================
Classifieur "simple" : StandardScaler + PCA + Random Forest (hyperparamètres fixes).
Split 80 % train / 15 % test / 5 % validation.
"""

import json

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .grammaire import analyser_phrase, calculer_statistiques
from .vectorisation import embedding_phrase


N_PCA = 50  # Composantes PCA à conserver (règle pratique : ~95% de la variance)


def charger_dataset(df: pd.DataFrame) -> tuple:
    """
    Extrait les features (X) et les labels (y) depuis un DataFrame.

    Pipeline appliqué :
      1. StandardScaler sur stats + embedding
      2. PCA(N_PCA) sur l'embedding uniquement

    Returns:
        X (n_phrases, N_PCA+2), y (n_phrases,), preprocessor (dict)
    """
    embeddings = np.array([json.loads(e) for e in df["embedding"]])
    stats      = df[["total_mots", "total_ponctuation"]].values
    y          = df["label"].values

    scaler_stats = StandardScaler()
    scaler_emb   = StandardScaler()
    stats_norm   = scaler_stats.fit_transform(stats)
    emb_norm     = scaler_emb.fit_transform(embeddings)

    n_comp     = min(N_PCA, emb_norm.shape[0] - 1, emb_norm.shape[1])
    pca        = PCA(n_components=n_comp, random_state=42)
    emb_reduit = pca.fit_transform(emb_norm)
    variance   = pca.explained_variance_ratio_.sum()

    X = np.hstack([stats_norm, emb_reduit])

    print(f"✅ Dataset chargé : {len(df)} phrases — {int(y.sum())} IA, {int((y==0).sum())} humain")
    print(f"   Features : 2 stats + {n_comp} PCA = {X.shape[1]} total")
    print(f"   Variance expliquée par la PCA : {variance:.1%}")

    preprocessor = {"scaler_stats": scaler_stats, "scaler_emb": scaler_emb, "pca": pca}
    return X, y, preprocessor


def splitter_dataset(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Divise le dataset en trois ensembles :
      - train      : 80% — entraîner le modèle
      - test       : 15% — évaluation finale (une seule fois)
      - validation :  5% — ajuster les hyperparamètres

    Returns:
        dict avec X_train, X_test, X_val, y_train, y_test, y_val
    """
    X_train, X_reste, y_train, y_reste = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_reste, y_reste, test_size=0.25, random_state=42, stratify=y_reste
    )

    print(f"✅ Split ({len(y)} phrases) : train={len(y_train)}, test={len(y_test)}, val={len(y_val)}")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_test":  X_test,  "y_test":  y_test,
        "X_val":   X_val,   "y_val":   y_val,
    }


def entrainer_modele(X_train: np.ndarray, y_train: np.ndarray,
                     X_val:   np.ndarray, y_val:   np.ndarray) -> RandomForestClassifier:
    """
    Entraîne un Random Forest sur le jeu d'entraînement et l'évalue sur la validation.

    Returns:
        Modèle Random Forest entraîné
    """
    modele = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        min_samples_leaf=1, random_state=42, n_jobs=-1,
    )
    modele.fit(X_train, y_train)

    score_train = modele.score(X_train, y_train)
    score_val   = modele.score(X_val, y_val)
    print(f"✅ Modèle entraîné  —  train={score_train:.2%}  val={score_val:.2%}")
    if score_train - score_val > 0.1:
        print("   ⚠️  Écart train/validation élevé — possible surapprentissage")

    return modele


def evaluer_modele(modele: RandomForestClassifier,
                   X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Évalue le modèle sur le jeu de test et affiche le rapport de classification."""
    y_pred = modele.predict(X_test)
    print(f"\n📊 Évaluation finale sur le jeu de test :")
    print(classification_report(y_test, y_pred, target_names=["humain", "IA"]))


def vectoriser_pour_prediction(phrase: str, preprocessor: dict) -> np.ndarray:
    """
    Calcule le vecteur de features d'une phrase en appliquant le même pipeline
    que charger_dataset() : normalisation + PCA.

    Returns:
        Vecteur numpy (N_PCA+2,) prêt pour le modèle
    """
    analyse    = analyser_phrase(phrase)
    stats_calc = calculer_statistiques(analyse)

    stats_brutes = np.array([[stats_calc["total_mots"], stats_calc["total_ponctuation"]]])
    emb_brut     = np.array(embedding_phrase(phrase)).reshape(1, -1)

    stats_norm   = preprocessor["scaler_stats"].transform(stats_brutes)
    emb_norm     = preprocessor["scaler_emb"].transform(emb_brut)
    emb_reduit   = preprocessor["pca"].transform(emb_norm)

    return np.hstack([stats_norm, emb_reduit])


def predire(phrase: str, modele: RandomForestClassifier, preprocessor: dict) -> dict:
    """
    Prédit si une phrase a été écrite par une IA ou par un humain.

    Returns:
        dict avec prediction ("IA" ou "humain"), label (1/0), confiance (0-1)
    """
    vecteur   = vectoriser_pour_prediction(phrase, preprocessor)
    label     = modele.predict(vecteur)[0]
    proba     = modele.predict_proba(vecteur)[0]
    confiance = proba[label]

    return {
        "prediction": "IA" if label == 1 else "humain",
        "label":      int(label),
        "confiance":  round(float(confiance), 4),
    }
