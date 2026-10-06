"""Algorithme génétique + entraînement + prédiction, avec de faux embeddings."""

import json
import random


from projet_ia.detection import Genome, algorithme_genetique, entrainer_meilleur
from projet_ia.detection.evolution import croiser, muter
from projet_ia.detection.genome import calculer_fitness
from projet_ia.detection.prediction import predire_depuis_fichier


def test_genome_reste_dans_ses_bornes():
    random.seed(1)
    for _ in range(200):
        g = muter(croiser(Genome.aleatoire(), Genome.aleatoire()), taux=1.0)
        assert 5 <= g.n_pca <= 200 and 50 <= g.n_estimators <= 500
        assert 100 <= g.n_camembert <= 768 and g.n_pca <= g.n_camembert
        assert g.use_embedding or g.use_total_mots or g.use_total_ponct


def test_genome_aller_retour_dict():
    g = Genome.aleatoire()
    assert Genome.from_dict(g.to_dict()).to_dict() == g.to_dict()


def test_fitness_favorise_precision_et_rapidite():
    rapide = Genome(); rapide.accuracy, rapide.temps_inference = 0.9, 1.0
    lent = Genome();   lent.accuracy,   lent.temps_inference = 0.9, 5.0
    calculer_fitness([rapide, lent])
    assert rapide.fitness > lent.fitness


def test_pipeline_complet_genetique_puis_prediction(
        tmp_path, dataset_phrases, faux_modeles_detection):
    sauvegarde, historique, pkl = (tmp_path / "g.json", tmp_path / "h.json",
                                   tmp_path / "sous" / "m.pkl")
    random.seed(0)
    genome = algorithme_genetique(
        str(dataset_phrases), taille_population=6, taille_elite=3, taille_batch=40,
        patience=1, max_generations=2,
        chemin_sauvegarde=str(sauvegarde), chemin_historique=str(historique))

    assert sauvegarde.exists() and json.loads(historique.read_text())
    assert genome.accuracy > 0.8

    entrainer_meilleur(genome, str(dataset_phrases), str(pkl))
    assert pkl.exists()

    ia = predire_depuis_fichier("Le robot produit encore une phrase.", pkl)
    humain = predire_depuis_fichier("Moi je raconte ma soirée comme ça", pkl)
    assert ia["prediction"] == "IA" and humain["prediction"] == "humain"
    assert 0.0 <= ia["confiance"] <= 1.0
