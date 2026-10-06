"""
detection/mlp.py
================
Classifieur MLP et ses variations.

Un `GenomeMLP` décrit une variante : architecture (nombre de couches, largeur,
forme constante ou pyramidale), activation, régularisation, optimisation, et
groupes de features utilisés (embedding, style, cohérence, perplexité). L'algorithme
génétique (evolution.py) cherche le meilleur génome.

L'entraînement se fait époque par époque : on suit la perte et l'accuracy de
validation, et on garde les meilleurs poids (arrêt anticipé).
"""

import copy
import random
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ACTIVATIONS = ("relu", "tanh", "logistic")
FORMES = ("constante", "pyramide")
BATCH_SIZES = (128, 256, 512, 1024, 2048)
BLOCS_TOUS = ("emb", "style", "coh", "ppl")


@dataclass
class GenomeMLP:
    # ── architecture ──
    n_couches: int = 2            # 1 à 4 couches cachées
    largeur: int = 256            # neurones de la 1re couche cachée (32 à 1024)
    forme: str = "constante"      # "constante" : même largeur ; "pyramide" : largeur ÷ 2 par couche
    activation: str = "relu"
    # ── régularisation / optimisation ──
    alpha: float = 1e-4           # pénalité L2
    lr: float = 1e-3              # taux d'apprentissage initial (Adam)
    batch_size: int = 512
    # ── features utilisées ──
    use_embedding: bool = True
    use_style: bool = True
    use_coherence: bool = False   # features de cohérence entre phrases
    use_ppl: bool = False         # features de perplexité (modèle de langue causal)
    # ── résultats (remplis par l'évaluation) ──
    accuracy: Optional[float] = None
    temps_entrainement: Optional[float] = None
    fitness: Optional[float] = None

    def couches(self) -> tuple:
        if self.forme == "pyramide":
            return tuple(max(16, self.largeur // (2 ** i)) for i in range(self.n_couches))
        return (self.largeur,) * self.n_couches

    def valider(self, blocs: tuple = BLOCS_TOUS) -> "GenomeMLP":
        """
        Ramène le génome dans ses bornes. `blocs` liste les groupes de features
        disponibles ("emb", "style", "coh", "ppl") ; les autres sont désactivés, et au moins
        un groupe reste actif.
        """
        self.n_couches = int(min(4, max(1, self.n_couches)))
        self.largeur = int(min(1024, max(32, self.largeur)))
        self.alpha = float(min(1e-1, max(1e-7, self.alpha)))
        self.lr = float(min(1e-2, max(1e-4, self.lr)))
        if self.batch_size not in BATCH_SIZES:
            self.batch_size = min(BATCH_SIZES, key=lambda b: abs(b - self.batch_size))
        if self.activation not in ACTIVATIONS:
            self.activation = "relu"
        if self.forme not in FORMES:
            self.forme = "constante"
        self.use_embedding = self.use_embedding and "emb" in blocs
        self.use_style = self.use_style and "style" in blocs
        self.use_coherence = self.use_coherence and "coh" in blocs
        self.use_ppl = self.use_ppl and "ppl" in blocs
        if not (self.use_embedding or self.use_style or self.use_coherence or self.use_ppl):
            premier = blocs[0]
            setattr(self, {"emb": "use_embedding", "style": "use_style",
                           "coh": "use_coherence", "ppl": "use_ppl"}[premier], True)
        return self

    @classmethod
    def aleatoire(cls, blocs: tuple = BLOCS_TOUS) -> "GenomeMLP":
        g = cls(
            n_couches=random.randint(1, 4),
            largeur=random.choice([32, 64, 128, 256, 512, 1024]),
            forme=random.choice(FORMES),
            activation=random.choice(ACTIVATIONS),
            alpha=10 ** random.uniform(-7, -1),
            lr=10 ** random.uniform(-4, -2),
            batch_size=random.choice(BATCH_SIZES),
            use_embedding=random.random() < 0.85,
            use_style=random.random() < 0.85,
            use_coherence=random.random() < 0.6,
            use_ppl=random.random() < 0.85,
        )
        return g.valider(blocs)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GenomeMLP":
        champs = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in champs})


def assembler_features(emb, style, coh, genome: GenomeMLP, ppl=None) -> np.ndarray:
    """Concatène les groupes de features choisis par le génome (float32)."""
    blocs = []
    if genome.use_embedding:
        blocs.append(np.asarray(emb, dtype=np.float32))
    if genome.use_style:
        blocs.append(np.asarray(style, dtype=np.float32))
    if genome.use_coherence:
        if coh is None:
            raise ValueError("Ce génome utilise la cohérence, mais elle n'est pas fournie.")
        blocs.append(np.asarray(coh, dtype=np.float32))
    if genome.use_ppl:
        if ppl is None:
            raise ValueError("Ce génome utilise la perplexité, mais elle n'est pas fournie.")
        blocs.append(np.asarray(ppl, dtype=np.float32))
    if not blocs:
        raise ValueError("Le génome n'utilise aucune feature.")
    return np.hstack(blocs)


def entrainer_mlp(genome: GenomeMLP, X_train, y_train, X_val, y_val,
                  epochs_max: int = 10, patience: int = 3, seed: int = 42) -> dict:
    """
    Entraîne un MLP décrit par `genome`, époque par époque.

    Les features sont standardisées (moyenne/écart-type du train). À chaque
    époque on mesure l'accuracy sur (X_val, y_val) ; on garde les meilleurs
    poids et on s'arrête après `patience` époques sans amélioration.

    Returns:
        dict : modele, scaler, accuracy_val, epoch_optimal, historique
               (perte d'entraînement et accuracy de validation par époque), temps
    """
    debut = time.perf_counter()
    scaler = StandardScaler().fit(X_train)
    Xt, Xv = scaler.transform(X_train), scaler.transform(X_val)

    clf = MLPClassifier(
        hidden_layer_sizes=genome.couches(), activation=genome.activation,
        solver="adam", alpha=genome.alpha, batch_size=genome.batch_size,
        learning_rate_init=genome.lr, shuffle=True, random_state=seed,
    )

    meilleure_acc, meilleurs_poids, epoch_opt, sans_amelioration = -1.0, None, 0, 0
    historique = []
    for epoch in range(1, epochs_max + 1):
        clf.partial_fit(Xt, y_train, classes=np.array([0, 1]))
        acc = float(clf.score(Xv, y_val))
        historique.append({"epoch": epoch, "loss_train": round(float(clf.loss_), 6),
                           "accuracy_val": round(acc, 4)})
        if acc > meilleure_acc + 1e-4:
            meilleure_acc, epoch_opt, sans_amelioration = acc, epoch, 0
            meilleurs_poids = (copy.deepcopy(clf.coefs_), copy.deepcopy(clf.intercepts_))
        else:
            sans_amelioration += 1
            if sans_amelioration >= patience:
                break

    clf.coefs_, clf.intercepts_ = meilleurs_poids
    return {"modele": clf, "scaler": scaler, "accuracy_val": meilleure_acc,
            "epoch_optimal": epoch_opt, "historique": historique,
            "temps": time.perf_counter() - debut}


def predire_proba(entraine: dict, X) -> np.ndarray:
    """Probabilité « IA » (classe 1) pour chaque ligne de X."""
    return entraine["modele"].predict_proba(entraine["scaler"].transform(X))[:, 1]


def seuil_pour_fpr(y, proba, fpr_cible: float) -> float:
    """
    Seuil de décision le plus bas tel qu'au plus `fpr_cible` des textes HUMAINS
    soient classés « IA » (taux de faux positifs) : c'est le seuil qui détecte le
    plus de textes IA sans dépasser ce taux d'accusations à tort.

    Avec k = fpr_cible × n_humains faux positifs tolérés, le seuil est placé juste
    au-dessus du (k+1)-ième score humain le plus élevé.
    """
    y, proba = np.asarray(y), np.asarray(proba, dtype=float)
    humains = np.sort(proba[y == 0])[::-1]                  # du plus « IA » au moins « IA »
    if len(humains) == 0:
        raise ValueError("Aucun texte humain : impossible de mesurer les faux positifs.")
    k = int(np.floor(fpr_cible * len(humains)))             # faux positifs tolérés
    if k >= len(humains):
        return 0.0
    return float(min(np.nextafter(humains[k], np.inf), 1.0))


def calculer_metriques(y, proba, seuil: float = 0.5) -> dict:
    """Accuracy, précision, rappel, F1, AUC, taux de faux positifs et matrice de
    confusion (classe positive = IA)."""
    y = np.asarray(y)
    pred = (np.asarray(proba) >= seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "seuil": round(float(seuil), 6),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "rappel": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "taux_faux_positifs": round(float(fp / (fp + tn)), 4) if fp + tn else None,
        "auc": round(float(roc_auc_score(y, proba)), 4) if len(set(y.tolist())) == 2 else None,
        "confusion": {"vrais_negatifs": int(tn), "faux_positifs": int(fp),
                      "faux_negatifs": int(fn), "vrais_positifs": int(tp)},
    }


def metriques_par_groupe(meta: pd.DataFrame, y, proba, colonne: str,
                         seuil: float = 0.5, min_n: int = 1) -> dict:
    """
    Accuracy par valeur de `colonne` (domaine, longueur, generateur…).

    Pour "generateur", seuls les textes IA sont considérés (les textes humains
    n'ont pas de générateur) : l'accuracy y est donc le taux de détection de
    chaque modèle de langue.
    """
    y, proba = np.asarray(y), np.asarray(proba)
    pred = (proba >= seuil).astype(int)
    masque = (y == 1) if colonne == "generateur" else np.ones(len(y), dtype=bool)
    resultat = {}
    for valeur, idx in meta[masque].groupby(colonne).groups.items():
        pos = meta.index.get_indexer(idx)
        if len(pos) >= min_n:
            resultat[str(valeur)] = {"n": int(len(pos)),
                                     "accuracy": round(float(np.mean(pred[pos] == y[pos])), 4)}
    return dict(sorted(resultat.items(), key=lambda kv: kv[1]["accuracy"]))
