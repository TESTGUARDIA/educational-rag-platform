import json
import random
import shutil

import numpy as np
import pytest

from conftest import faux_encodeur, faux_ppl
from projet_ia import config
from projet_ia.detection import (COMPOSANTS, GenomeMLP, algorithme_genetique, construire_cache,
                                 entrainer_final, predire_textes)
from projet_ia.detection.composants import BLOCS, genome_par_defaut
from projet_ia.detection.evolution import MemoireAntibiais, calculer_fitness, croiser, muter
from projet_ia.detection.mlp import calculer_metriques, entrainer_mlp, seuil_pour_fpr


@pytest.mark.parametrize("composant", COMPOSANTS)
def test_genome_toujours_dans_ses_bornes(composant):
    random.seed(1)
    blocs = BLOCS[composant]
    for _ in range(300):
        g = muter(croiser(GenomeMLP.aleatoire(blocs), GenomeMLP.aleatoire(blocs), blocs), 1.0, blocs)
        assert 1 <= g.n_couches <= 4 and 32 <= g.largeur <= 1024
        assert 1e-7 <= g.alpha <= 1e-1 and 1e-4 <= g.lr <= 1e-2
        assert g.use_embedding or g.use_style or g.use_coherence or g.use_ppl
        assert (not g.use_embedding) or "emb" in blocs      # jamais un bloc indisponible
        assert (not g.use_coherence) or "coh" in blocs
        assert (not g.use_style) or "style" in blocs
        assert (not g.use_ppl) or "ppl" in blocs
        assert len(g.couches()) == g.n_couches


def test_genomes_par_defaut_respectent_les_blocs():
    for c in COMPOSANTS:
        g = genome_par_defaut(c)
        assert g.to_dict() == GenomeMLP.from_dict(g.to_dict()).valider(BLOCS[c]).to_dict()


def test_genome_forme_et_aller_retour_dict():
    assert GenomeMLP(n_couches=3, largeur=256, forme="pyramide").couches() == (256, 128, 64)
    assert GenomeMLP(n_couches=2, largeur=64, forme="constante").couches() == (64, 64)
    g = GenomeMLP.aleatoire()
    assert GenomeMLP.from_dict(g.to_dict()).to_dict() == g.to_dict()


def test_memoire_antibiais_lots_disjoints():
    m = MemoireAntibiais(fenetre=2, seed=0)
    l1, l2 = m.tirer_lot(1000, 300), m.tirer_lot(1000, 300)
    assert len(l1) == len(l2) == 300 and not set(l1) & set(l2)
    assert len(m.tirer_lot(100, 500)) == 100                # lot plus grand que le dataset


def test_fitness_favorise_le_plus_rapide_a_precision_egale():
    rapide, lent = GenomeMLP(), GenomeMLP()
    rapide.accuracy = lent.accuracy = 0.9
    rapide.temps_entrainement, lent.temps_entrainement = 1.0, 9.0
    calculer_fitness([rapide, lent])
    assert rapide.fitness > lent.fitness


def _individu(accuracy, temps):
    g = GenomeMLP()
    g.accuracy, g.temps_entrainement = accuracy, temps
    return g


def test_fitness_la_vitesse_ne_fait_que_departager():
    # Cas réel du 1er entraînement : +1 point d'accuracy pour 7× plus de temps -> le précis gagne
    precis, rapide = _individu(0.8978, 2.8), _individu(0.8882, 0.4)
    calculer_fitness([precis, rapide])
    assert precis.fitness > rapide.fitness
    # +0,2 point pour 10× plus de temps : pas assez, le rapide gagne
    lent, vif = _individu(0.902, 10.0), _individu(0.900, 1.0)
    calculer_fitness([lent, vif])
    assert vif.fitness > lent.fitness
    # 10× plus lent = exactement poids_rapidite d'accuracy perdue
    a, b = _individu(0.9, 1.0), _individu(0.9, 10.0)
    calculer_fitness([a, b], poids_rapidite=0.005)
    assert a.fitness - b.fitness == pytest.approx(0.005)


def test_fitness_comparable_entre_generations():
    # Avec un temps de référence fixe, la fitness d'un individu ne dépend pas des autres
    g1 = [_individu(0.9, 2.0), _individu(0.8, 0.1)]
    g2 = [_individu(0.9, 2.0), _individu(0.85, 50.0)]
    ref = calculer_fitness(g1)
    assert calculer_fitness(g2, temps_reference=ref) == ref
    assert g1[0].fitness == g2[0].fitness


def test_entrainement_mlp_courbe_et_arret_anticipe():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 10)); y = (X[:, 0] + X[:, 1] > 0).astype(int)
    res = entrainer_mlp(GenomeMLP(n_couches=1, largeur=32, batch_size=32, lr=1e-2), X[:400],
                        y[:400], X[400:], y[400:], epochs_max=30, patience=2)
    assert res["accuracy_val"] > 0.9
    assert 1 <= len(res["historique"]) <= 30
    assert set(res["historique"][0]) == {"epoch", "loss_train", "accuracy_val"}


def test_metriques():
    m = calculer_metriques([1, 1, 0, 0], [0.9, 0.4, 0.2, 0.6])
    assert m["accuracy"] == 0.5 and m["confusion"]["vrais_positifs"] == 1
    assert m["taux_faux_positifs"] == 0.5 and m["seuil"] == 0.5


def test_seuil_borne_les_faux_positifs():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000, dtype=int), np.ones(1000, dtype=int)]
    proba = np.r_[rng.beta(2, 5, 1000), rng.beta(5, 2, 1000)]   # classes qui se chevauchent
    for cible in (0.0, 0.01, 0.05, 0.2):
        s = seuil_pour_fpr(y, proba, cible)
        fpr = np.mean(proba[y == 0] >= s)
        assert fpr <= cible
        assert np.mean(proba[y == 0] >= np.nextafter(s, 0)) > cible or s == 0.0  # le plus bas possible
    # plus on tolère de faux positifs, plus le seuil baisse et plus on détecte d'IA
    assert seuil_pour_fpr(y, proba, 0.01) > seuil_pour_fpr(y, proba, 0.2)
    assert seuil_pour_fpr([0, 0, 1], [0.1, 0.2, 0.9], 1.0) == 0.0
    with pytest.raises(ValueError):
        seuil_pour_fpr([1, 1], [0.3, 0.8], 0.01)


def test_pipeline_complet_sans_toucher_au_test_pendant_la_recherche(
        tmp_path, dataset_synthetique):
    cache = tmp_path / "cache"
    construire_cache(dataset_synthetique, cache, taille_chunk=200, encodeur=faux_encodeur, calculateur_ppl=faux_ppl)

    # Le jeu de test est mis hors de portée : la recherche génétique ne doit pas en avoir besoin.
    cache_test = tmp_path / "test_cache_cache"; cache_test.mkdir()
    for f in cache.glob("*test*"):
        shutil.move(str(f), cache_test / f.name)

    random.seed(0)
    genomes = {}
    for c in COMPOSANTS:
        genomes[c] = algorithme_genetique(
            cache, composant=c, taille_population=4, taille_elite=2, taille_lot_train=300,
            taille_val=100, epochs_individu=3, patience=1, max_generations=2,
            chemin_sauvegarde=tmp_path / f"g_{c}.json", chemin_historique=tmp_path / f"h_{c}.json")
        assert genomes[c].accuracy > 0.6
        assert json.loads((tmp_path / f"g_{c}.json").read_text())["accuracy"] == genomes[c].accuracy
        assert json.loads((tmp_path / f"h_{c}.json").read_text())

    for f in cache_test.iterdir():                          # le test réapparaît pour l'évaluation finale
        shutil.move(str(f), cache / f.name)

    rapport = entrainer_final(genomes, cache, tmp_path / "m" / "modele.pkl",
                              tmp_path / "r" / "rapport.json", epochs_max=8)

    assert rapport["tailles"] == {"val": 100, "test": 100}
    assert rapport["test"]["accuracy"] > 0.8
    assert set(rapport["composants"]) == set(COMPOSANTS)
    assert all("test" in d and d["historique"] for d in rapport["composants"].values())
    fu = rapport["fusion"]
    assert set(fu["poids_ia"]) == set(fu["poids_humain"]) and "biais" in fu
    seuil = rapport["seuil"]["valeur"]
    assert rapport["seuil"]["fpr_cible"] == config.FPR_CIBLE_DETECTION
    assert fu["validation"]["taux_faux_positifs"] <= config.FPR_CIBLE_DETECTION
    assert rapport["test"]["seuil"] == round(seuil, 6) and rapport["test_seuil_0_5"]["seuil"] == 0.5
    assert rapport["test_niveau_phrase"]["n"] == 100 * 5     # toutes les phrases du test
    for cle in ("test_par_domaine", "test_par_longueur", "test_detection_par_generateur"):
        assert rapport[cle]
    assert (tmp_path / "r" / "rapport.json").exists()

    ia, humain = dataset_synthetique[dataset_synthetique.split == "test"] \
        .sort_values("label", ascending=False).texte.iloc[[0, -1]]
    res = predire_textes([ia, humain, "Une phrase."], chemin_modele=tmp_path / "m" / "modele.pkl",
                         encodeur=faux_encodeur, calculateur_ppl=faux_ppl)
    assert res[0]["prediction"] == "IA" and res[1]["prediction"] == "humain"
    assert res[0]["seuil"] == round(seuil, 4)                # seuil appris, sauvegardé dans le .pkl
    force = predire_textes([humain], chemin_modele=tmp_path / "m" / "modele.pkl",
                           encodeur=faux_encodeur, calculateur_ppl=faux_ppl, seuil=0.0)
    assert force[0]["prediction"] == "IA" and force[0]["seuil"] == 0.0
    assert len(res[0]["phrases"]) == 5 and set(res[0]["composants"]) == {
        "paragraphe", "interphrases", "phrases_moy"}
    assert "avertissement" not in res[0] and "avertissement" in res[2]


def test_modeles_par_langue_et_routage_automatique(tmp_path, dataset_synthetique, monkeypatch):
    monkeypatch.setattr(config, "DOSSIER_MODELS", tmp_path / "models")
    monkeypatch.setattr(config, "DOSSIER_DATA", tmp_path / "data")

    cache_en = config.CheminsDetection("en").cache
    construire_cache(dataset_synthetique, cache_en, langue="en", taille_chunk=300,
                     encodeur=faux_encodeur, calculateur_ppl=faux_ppl)
    rapport = entrainer_final(None, cache_en, epochs_max=5)       # chemins déduits de la langue
    assert rapport["langue"] == "en"
    assert config.CheminsDetection("en").modele.exists()
    assert config.CheminsDetection("en").rapport.exists()

    anglais = ("The robot was sitting on the table and it was not happy with the situation. "
               "It had been there for a long time and the others were not there. "
               "This was the end of the story for them.")
    francais = ("Le robot dormait sur la table et il ne voulait pas se lever pour nous. "
                "Il était là depuis longtemps et les autres ne sont pas venus. "
                "Ce fut la fin de cette histoire pour eux.")

    r = predire_textes([anglais], encodeur=faux_encodeur, calculateur_ppl=faux_ppl)[0]       # langue détectée -> modèle en
    assert r["langue"] == "en" and "avertissement" not in r

    with pytest.raises(FileNotFoundError, match="fr"):             # pas de modèle français
        predire_textes([francais], encodeur=faux_encodeur, calculateur_ppl=faux_ppl)

    shutil.copytree(config.CheminsDetection("en").dossier_modeles,
                    config.CheminsDetection("multi").dossier_modeles)   # modèle de repli
    r = predire_textes([francais], encodeur=faux_encodeur, calculateur_ppl=faux_ppl)[0]
    assert "multilingue" in r["avertissement"]
    assert predire_textes([anglais], langue="en", encodeur=faux_encodeur, calculateur_ppl=faux_ppl)[0]["langue"] == "en"
