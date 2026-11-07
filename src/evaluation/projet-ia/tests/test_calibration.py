import json

import numpy as np
import pandas as pd

from projet_ia.calibration import calibrer


def _meta_dataset(chemin, n=200, seed=0):
    rng = np.random.default_rng(seed)
    nli = rng.uniform(0, 1, n)
    sim = rng.uniform(0, 1, n)
    label = ((0.6 * nli + 0.4 * sim + rng.normal(0, 0.05, n)) > 0.5).astype(int)
    df = pd.DataFrame({
        "question": "q", "reponse": "r", "correction": "c", "label": label,
        "caractere_question": rng.choice(["fermée", "ouverte"], n),
        "nli": nli, "similarite_ponderee": sim, "sens_global": rng.uniform(0, 1, n),
        "penalite_neg": 0.0, "penalite_entites": 0.0, "penalite_concept": 0.0,
    })
    df.to_csv(chemin, index=False)


def test_calibrer_genere_des_poids_valides(tmp_path):
    csv, sortie = tmp_path / "meta.csv", tmp_path / "poids.json"
    _meta_dataset(csv)

    calibrer(str(csv), str(sortie), epochs=800, k_fold=3)

    data = json.loads(sortie.read_text(encoding="utf-8"))
    for cle in ["nli", "similarite_ponderee", "sens_global", "biais",
                "q_fermee", "q_semi_ouverte", "q_ouverte"]:
        assert cle in data["poids"]
    assert data["poids"]["nli"] > 0                      # le NLI doit aider à prédire "correct"
    assert data["metriques_entrainement"]["f1"] > 0.85
    assert data["n_exemples_entrainement"] == 200


def test_score_calibre_utilise_les_poids_fournis():
    from projet_ia.verification import score_calibre

    poids = {"nli": 10.0, "biais": -5.0}
    haut = score_calibre({"nli": 1.0}, poids=poids)
    bas = score_calibre({"nli": 0.0}, poids=poids)
    assert haut > 0.99 and bas < 0.01
