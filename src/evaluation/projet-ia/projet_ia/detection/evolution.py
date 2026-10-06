"""
detection/evolution.py
======================
Algorithme génétique : recherche du meilleur MLP (GenomeMLP) pour un composant
(paragraphe, phrase ou interphrases — voir composants.py).

Protocole (conçu pour que le jeu de test reste vierge) :
  * Les features viennent du CACHE (aucun ré-encodage).
  * Seuls les splits `train` et `val` sont lus. Le split `test` n'est JAMAIS
    ouvert ici : il ne sert qu'une fois, à la fin (entrainement.entrainer_final).
  * `val` : sous-échantillon FIXE → les fitness de toutes les générations sont
    comparables entre elles.
  * `train` : à chaque génération, un nouveau lot de textes (MemoireAntibiais)
    pour éviter de sur-optimiser sur un lot particulier.

  fitness = accuracy_val × (1 − poids_rapidite) + rapidité × poids_rapidite
  (rapidité = temps d'entraînement normalisé dans la population : 1 = le plus rapide)
"""

import json
import random
from collections import deque
from copy import deepcopy

import numpy as np

from .. import config
from ..common.fichiers import creer_dossier_parent
from .cache import lire_info_cache
from .composants import BLOCS, charger_donnees
from .mlp import (ACTIVATIONS, BATCH_SIZES, BLOCS_TOUS, FORMES, GenomeMLP,
                  assembler_features, entrainer_mlp)


def muter(genome: GenomeMLP, taux: float = 0.2, blocs: tuple = BLOCS_TOUS) -> GenomeMLP:
    """Mutation : chaque gène change avec la probabilité `taux`."""
    g = deepcopy(genome)
    if random.random() < taux:
        g.n_couches = g.n_couches + random.choice([-1, 1])
    if random.random() < taux:
        g.largeur = int(g.largeur * random.choice([0.5, 0.75, 1.5, 2.0]))
    if random.random() < taux:
        g.forme = random.choice(FORMES)
    if random.random() < taux:
        g.activation = random.choice(ACTIVATIONS)
    if random.random() < taux:
        g.alpha = g.alpha * 10 ** random.uniform(-1, 1)
    if random.random() < taux:
        g.lr = g.lr * 10 ** random.uniform(-0.5, 0.5)
    if random.random() < taux:
        g.batch_size = random.choice(BATCH_SIZES)
    if random.random() < taux:
        g.use_embedding = not g.use_embedding
    if random.random() < taux:
        g.use_style = not g.use_style
    if random.random() < taux:
        g.use_coherence = not g.use_coherence
    g.accuracy = g.temps_entrainement = g.fitness = None
    return g.valider(blocs)


def croiser(a: GenomeMLP, b: GenomeMLP, blocs: tuple = BLOCS_TOUS) -> GenomeMLP:
    """Croisement uniforme : chaque gène vient de l'un ou l'autre parent."""
    enfant = GenomeMLP()
    for champ in ("n_couches", "largeur", "forme", "activation", "alpha", "lr",
                  "batch_size", "use_embedding", "use_style", "use_coherence"):
        setattr(enfant, champ, getattr(random.choice([a, b]), champ))
    return enfant.valider(blocs)


def reproduire(elite: list, taille_population: int, taux_mutation: float,
               blocs: tuple = BLOCS_TOUS) -> list:
    """Nouvelle génération : l'élite est conservée, le reste est créé par croisement + mutation."""
    nouvelle = [deepcopy(g) for g in elite]
    while len(nouvelle) < taille_population:
        a, b = random.sample(elite, 2) if len(elite) >= 2 else (elite[0], elite[0])
        nouvelle.append(muter(croiser(a, b, blocs), taux_mutation, blocs))
    return nouvelle


class MemoireAntibiais:
    """
    Évite de réévaluer sur les mêmes exemples de train d'une génération à l'autre :
    chaque lot tiré évite les `fenetre` derniers lots (tant qu'il reste assez
    d'exemples inédits).
    """

    def __init__(self, fenetre: int = 3, seed: int = 0):
        self.lots = deque(maxlen=fenetre)
        self.rng = np.random.default_rng(seed)

    def tirer_lot(self, n_total: int, taille: int) -> np.ndarray:
        if taille >= n_total:
            lot = np.arange(n_total)
        else:
            vus = np.zeros(n_total, dtype=bool)
            for ancien in self.lots:
                vus[ancien] = True
            inedits = np.flatnonzero(~vus)
            if len(inedits) >= taille:
                lot = self.rng.choice(inedits, taille, replace=False)
            else:
                anciens = self.rng.choice(np.flatnonzero(vus), taille - len(inedits), replace=False)
                lot = np.concatenate([inedits, anciens])
        self.lots.append(lot)
        return np.sort(lot)


def calculer_fitness(population: list, poids_rapidite: float = 0.1) -> None:
    """Calcule genome.fitness pour toute la population (modifie en place)."""
    temps = np.array([g.temps_entrainement for g in population], dtype=float)
    ecart = temps.max() - temps.min()
    for g, t in zip(population, temps):
        rapidite = 1.0 - (t - temps.min()) / ecart if ecart > 0 else 1.0
        g.fitness = g.accuracy * (1 - poids_rapidite) + rapidite * poids_rapidite


def _sel(tableau, idx):
    """tableau[idx], ou None si le composant n'a pas ce bloc de features."""
    return None if tableau is None else tableau[idx]


def algorithme_genetique(
    dossier_cache,
    composant: str = "paragraphe",
    taille_population: int = 30,
    taille_elite: int = 10,
    taille_lot_train: int = 50_000,
    taille_val: int = 20_000,
    epochs_individu: int = 5,
    patience: int = 5,
    max_generations: int = 20,
    taux_mutation_init: float = 0.3,
    taux_mutation_min: float = 0.05,
    fenetre_memoire: int = 3,
    poids_rapidite: float = 0.1,
    max_exemples_charges: int | None = 500_000,
    chemin_sauvegarde=None,
    chemin_historique=None,
    seed: int = 42,
    entraineur=entrainer_mlp,
) -> GenomeMLP:
    """
    Lance l'évolution pour un composant et retourne le meilleur GenomeMLP (aussi
    sauvegardé en JSON).

    Args:
        composant            : "paragraphe", "phrase" ou "interphrases"
        taille_lot_train     : exemples de train utilisés pour entraîner chaque individu
        taille_val           : taille du sous-échantillon fixe de validation
        epochs_individu      : époques maximum par individu (court : on compare des génomes)
        patience             : générations sans amélioration avant arrêt
        max_exemples_charges : composant "phrase" : nombre maximum de phrases chargées en mémoire
        chemin_sauvegarde / chemin_historique : par défaut dans models/<langue>/
        entraineur           : fonction d'entraînement (par défaut entrainer_mlp)
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)
    blocs = BLOCS[composant]
    chemins = config.CheminsDetection(lire_info_cache(dossier_cache)["langue"])
    chemin_sauvegarde = chemin_sauvegarde or chemins.genome(composant)
    chemin_historique = chemin_historique or chemins.historique(composant)

    print(f"[{composant}] Chargement des features (train, val)...")
    train = charger_donnees(dossier_cache, composant, "train", max_exemples_charges, seed)
    val = charger_donnees(dossier_cache, composant, "val", None, seed)
    idx_val = np.sort(rng.choice(len(val["y"]), min(taille_val, len(val["y"])), replace=False))
    y_val = val["y"][idx_val]
    n_train = len(train["y"])
    print(f"  train : {n_train:,} exemples | validation (fixe) : {len(idx_val):,} exemples")

    memoire = MemoireAntibiais(fenetre_memoire, seed)
    population = [GenomeMLP.aleatoire(blocs) for _ in range(taille_population)]
    meilleur, historique = None, []
    sans_amelioration, taux = 0, taux_mutation_init

    for generation in range(1, max_generations + 1):
        idx = memoire.tirer_lot(n_train, taille_lot_train)
        y_t = train["y"][idx]
        cache_features = {}                                   # (emb, style, coh) -> matrices

        for g in population:
            cle = (g.use_embedding, g.use_style, g.use_coherence)
            if cle not in cache_features:
                cache_features[cle] = (
                    assembler_features(_sel(train["emb"], idx), _sel(train["style"], idx),
                                       _sel(train["coh"], idx), g),
                    assembler_features(_sel(val["emb"], idx_val), _sel(val["style"], idx_val),
                                       _sel(val["coh"], idx_val), g),
                )
            X_t, X_v = cache_features[cle]
            res = entraineur(g, X_t, y_t, X_v, y_val, epochs_max=epochs_individu,
                             patience=3, seed=seed)
            g.accuracy, g.temps_entrainement = res["accuracy_val"], res["temps"]

        calculer_fitness(population, poids_rapidite)
        population.sort(key=lambda g: g.fitness, reverse=True)
        courant = population[0]
        print(f"[{composant}] Génération {generation:>3} | fitness {courant.fitness:.4f} | "
              f"acc val {courant.accuracy:.4f} | {courant.couches()} {courant.activation} | "
              f"emb={courant.use_embedding} style={courant.use_style} coh={courant.use_coherence}")

        historique.append({"generation": generation, "fitness": round(courant.fitness, 5),
                           "accuracy_val": round(courant.accuracy, 5),
                           "taux_mutation": round(taux, 4), "genome": courant.to_dict()})

        if meilleur is None or courant.fitness > meilleur.fitness + 1e-4:
            meilleur, sans_amelioration = deepcopy(courant), 0
            creer_dossier_parent(chemin_sauvegarde)
            with open(chemin_sauvegarde, "w") as f:
                json.dump(meilleur.to_dict(), f, indent=2)
        else:
            sans_amelioration += 1
            if sans_amelioration >= patience:
                print(f"Arrêt : {patience} générations sans amélioration.")
                break

        taux = max(taux_mutation_min, taux * 0.9)
        population = reproduire(population[:taille_elite], taille_population, taux, blocs)

    creer_dossier_parent(chemin_historique)
    with open(chemin_historique, "w") as f:
        json.dump(historique, f, indent=2)
    print(f"[{composant}] Meilleur génome sauvegardé : {chemin_sauvegarde}")
    return meilleur
