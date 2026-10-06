"""
detection/genome.py
===================
Génome (hyperparamètres d'un individu), vectorisation selon un génome,
évaluation par validation croisée et calcul de la fitness.
"""

import time
from dataclasses import dataclass, asdict
from typing import Optional
import random

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from ..common.embeddings import embedding_cls


@dataclass
class Genome:
    """
    Représente les hyperparamètres d'un individu dans la population génétique.

    Paramètres optimisables :
      n_pca          : [5, 200]      — composantes PCA sur l'embedding
      n_estimators   : [50, 500]     — arbres du Random Forest
      max_depth      : [3, 30] | None — profondeur max (None = illimitée)
      use_total_mots : bool          — inclure le nb de mots comme feature
      use_total_ponct: bool          — inclure la ponctuation comme feature
      use_embedding  : bool          — inclure l'embedding CamemBERT
      n_camembert    : [100, 768]    — dimensions CamemBERT utilisées avant PCA
    """
    n_pca:           int           = 50
    n_estimators:    int           = 200
    max_depth:       Optional[int] = 10
    use_total_mots:  bool          = True
    use_total_ponct: bool          = True
    use_embedding:   bool          = True
    n_camembert:     int           = 768

    # Scores (non transmis lors de la mutation)
    fitness:         float = 0.0
    accuracy:        float = 0.0
    temps_inference: float = 0.0

    def valider(self) -> "Genome":
        """Force toutes les valeurs dans leurs plages autorisées."""
        self.n_pca        = max(5, min(200, self.n_pca))
        self.n_estimators = max(50, min(500, self.n_estimators))
        self.n_camembert  = max(100, min(768, self.n_camembert))
        if self.max_depth is not None:
            self.max_depth = max(3, min(30, self.max_depth))
        if not any([self.use_total_mots, self.use_total_ponct, self.use_embedding]):
            self.use_embedding = True
        if self.use_embedding:
            self.n_pca = min(self.n_pca, self.n_camembert)
        return self

    def n_features(self) -> int:
        """Retourne le nombre total de features du vecteur final."""
        n = (1 if self.use_total_mots else 0) + (1 if self.use_total_ponct else 0)
        if self.use_embedding:
            n += self.n_pca
        return n

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        g = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        return g.valider()

    @classmethod
    def aleatoire(cls) -> "Genome":
        """Crée un génome avec des hyperparamètres aléatoires."""
        n_cam = random.randint(100, 768)
        n_pca = random.randint(5, min(200, n_cam))
        return cls(
            n_pca           = n_pca,
            n_estimators    = random.randint(50, 500),
            max_depth       = random.choice([None] + list(range(3, 31))),
            use_total_mots  = random.random() > 0.2,
            use_total_ponct = random.random() > 0.2,
            use_embedding   = True,
            n_camembert     = n_cam,
        ).valider()


def embedding_camembert(phrase: str, n_dims: int = 768) -> np.ndarray:
    """Encode une phrase avec CamemBERT et retourne les n_dims premières dimensions."""
    return embedding_cls(phrase)[:n_dims]


def vectoriser_phrase_genome(phrase: str, genome: Genome,
                              total_mots: int, total_ponct: int) -> np.ndarray:
    """Construit le vecteur de features d'une phrase selon un génome."""
    features = []
    if genome.use_total_mots:  features.append(float(total_mots))
    if genome.use_total_ponct: features.append(float(total_ponct))
    if genome.use_embedding:
        features.extend(embedding_camembert(phrase, genome.n_camembert).tolist())
    return np.array(features)


def construire_matrice_genome(phrases: list[str], labels: list[int],
                               stats: list[dict], genome: Genome) -> tuple:
    """
    Construit la matrice X et le vecteur y pour un batch de phrases.
    Applique StandardScaler + PCA selon les paramètres du génome.

    Returns:
        (X_reduit, y, preprocessor)
    """
    X_raw = np.array([
        vectoriser_phrase_genome(ph, genome, st["total_mots"], st["total_ponct"])
        for ph, st in zip(phrases, stats)
    ])
    y = np.array(labels)

    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_raw)

    pca = None
    if genome.use_embedding:
        n_stats  = (1 if genome.use_total_mots else 0) + (1 if genome.use_total_ponct else 0)
        X_stats  = X_norm[:, :n_stats]
        X_emb    = X_norm[:, n_stats:]
        n_comp   = min(genome.n_pca, X_emb.shape[1], X_emb.shape[0] - 1)
        pca      = PCA(n_components=n_comp, random_state=42)
        X_reduit = np.hstack([X_stats, pca.fit_transform(X_emb)]) if n_stats > 0 else pca.fit_transform(X_emb)
    else:
        X_reduit = X_norm

    return X_reduit, y, {"scaler": scaler, "pca": pca, "genome": genome}


def evaluer_individu(genome: Genome, phrases: list[str],
                     labels: list[int], stats: list[dict]) -> Genome:
    """
    Évalue un génome sur un batch de phrases via cross-validation.
    Met à jour genome.accuracy, genome.temps_inference, genome.fitness.
    """
    try:
        t0 = time.perf_counter()
        X, y, _ = construire_matrice_genome(phrases, labels, stats, genome)

        if X.shape[1] == 0 or X.shape[0] < 5:
            genome.fitness = 0.0
            return genome

        rf = RandomForestClassifier(
            n_estimators=genome.n_estimators, max_depth=genome.max_depth,
            random_state=42, n_jobs=-1,
        )
        n_folds = min(5, min(np.bincount(y.astype(int))))
        if n_folds < 2:
            genome.fitness = 0.0
            return genome

        scores = cross_val_score(rf, X, y, cv=n_folds, scoring="accuracy", n_jobs=-1)
        genome.accuracy        = round(float(scores.mean()), 4)
        genome.temps_inference = round(time.perf_counter() - t0, 4)
        genome.fitness         = genome.accuracy  # normalisé dans calculer_fitness()

    except Exception:
        genome.fitness = 0.0
        genome.accuracy = 0.0
        genome.temps_inference = 999.0

    return genome


def calculer_fitness(population: list[Genome]) -> list[Genome]:
    """
    Normalise les temps d'inférence et calcule la fitness finale :
      fitness = accuracy × 0.70 + rapidité_normalisée × 0.30
    """
    temps   = [g.temps_inference for g in population if g.temps_inference < 999]
    if not temps:
        return population

    t_min   = min(temps)
    t_range = max(temps) - t_min or 1.0

    for g in population:
        rapidite = 0.0 if g.temps_inference >= 999 else 1.0 - (g.temps_inference - t_min) / t_range
        g.fitness = round(g.accuracy * 0.70 + rapidite * 0.30, 4)

    return population
