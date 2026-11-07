import numpy as np

from projet_ia.calibration.regression import sigmoid
from projet_ia.detection.fusion import (NOMS_ENTREES, agreger_phrases, assembler_entrees,
                                        entrainer_fusion, logit, predire_fusion)


def test_logit_est_l_inverse_de_la_sigmoide():
    p = np.array([0.01, 0.3, 0.5, 0.9, 0.999])
    assert np.allclose(sigmoid(logit(p)), p, atol=1e-5)
    assert np.isfinite(logit([0.0, 1.0])).all()               # bornes sans infini


def test_agregation_des_phrases():
    z = [1.0, 3.0, -2.0, 5.0]
    agg = agreger_phrases(z, idx_texte=[0, 0, 0, 2], n_textes=4)
    assert agg["phrases_moy"].tolist() == [2.0 / 3, 0.0, 5.0, 0.0]
    assert agg["phrases_max"].tolist() == [3.0, 0.0, 5.0, 0.0]
    assert agg["phrases_min"].tolist() == [-2.0, 0.0, 5.0, 0.0]   # textes 1 et 3 sans phrase : neutre


def _donnees(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    informatif = (2 * y - 1) * 2.0 + rng.normal(0, 1.5, n)
    bruit = lambda: rng.normal(0, 2, n)
    return y, {"paragraphe": informatif, "interphrases": bruit(), "phrases_moy": bruit(),
               "phrases_max": bruit(), "phrases_min": bruit()}


def test_fusion_donne_du_poids_au_composant_informatif():
    y, z = _donnees()
    fusion = entrainer_fusion(assembler_entrees(z), y, epochs=1500)
    poids = {n: abs(fusion["poids_ia"][n]) + abs(fusion["poids_humain"][n]) for n in NOMS_ENTREES}
    assert poids["paragraphe"] > 5 * max(v for n, v in poids.items() if n != "paragraphe")
    p = predire_fusion(fusion, assembler_entrees(z))
    assert np.mean((p >= 0.5) == y) > 0.85
    cout = [c["cout"] for c in fusion["historique_cout"]]
    assert cout[-1] < cout[0]                                  # la descente de gradient converge


def test_poids_ia_et_humain_peuvent_differer():
    """Un composant fiable seulement quand il crie « IA » : poids_ia fort, poids_humain faible."""
    rng = np.random.default_rng(1)
    n = 6000
    y = rng.integers(0, 2, n)
    alerte = rng.random(n) < 0.5
    z_a = np.where((y == 1) & alerte, 4.0, rng.normal(-1.0, 0.3, n))   # négatif = non informatif
    z = {nom: rng.normal(0, 0.1, n) for nom in NOMS_ENTREES}
    z["paragraphe"] = z_a
    fusion = entrainer_fusion(assembler_entrees(z), y, epochs=1500)
    assert fusion["poids_ia"]["paragraphe"] > 1.0
    assert abs(fusion["poids_humain"]["paragraphe"]) < 0.5 * fusion["poids_ia"]["paragraphe"]


def test_predire_fusion_est_une_somme_ponderee_de_logits():
    fusion = {"noms": list(NOMS_ENTREES), "biais": -0.5,
              "poids_ia": {n: 1.0 for n in NOMS_ENTREES},
              "poids_humain": {n: 2.0 for n in NOMS_ENTREES}}
    Z = np.array([[1.0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0]])
    attendu = sigmoid(np.array([1.0 - 0.5, 2.0 * -1.0 - 0.5]))
    assert np.allclose(predire_fusion(fusion, Z), attendu)
