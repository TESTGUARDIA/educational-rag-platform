import pytest

from conftest import DIM_FAUX, faux_encodeur, faux_ppl
from projet_ia.detection.cache import charger_phrases, charger_split, construire_cache
from projet_ia.detection.style import (N_FEATURES_COHERENCE, N_FEATURES_PHRASE,
                                       N_FEATURES_STYLE)


class Compteur:
    def __init__(self):
        self.appels = 0

    def __call__(self, textes):
        self.appels += 1
        return faux_encodeur(textes)


def test_cache_et_relecture(tmp_path, dataset_synthetique):
    construire_cache(dataset_synthetique, tmp_path, taille_chunk=150, encodeur=faux_encodeur, calculateur_ppl=faux_ppl)
    train = charger_split(tmp_path, "train")
    assert train["emb"].shape == (400, DIM_FAUX) and train["emb"].dtype.name == "float16"
    assert train["style"].shape == (400, N_FEATURES_STYLE)
    assert train["coh"].shape == (400, N_FEATURES_COHERENCE)
    assert len(train["y"]) == len(train["meta"]) == 400
    assert charger_split(tmp_path, "test")["y"].shape == (100,)


def test_une_seule_passe_dans_l_encodeur_par_bloc(tmp_path, dataset_synthetique):
    compteur = Compteur()
    construire_cache(dataset_synthetique, tmp_path, taille_chunk=200, encodeur=compteur, calculateur_ppl=faux_ppl)
    assert compteur.appels == 4               # 4 blocs (train 2, val 1, test 1) = 4 passes, pas plus


def test_niveau_phrase(tmp_path, dataset_synthetique):
    construire_cache(dataset_synthetique, tmp_path, taille_chunk=150, encodeur=faux_encodeur, calculateur_ppl=faux_ppl)

    train = charger_phrases(tmp_path, "train")             # 3 phrases gardées par paragraphe
    assert train["emb"].shape == (400 * 3, DIM_FAUX)
    assert train["style"].shape == (1200, N_FEATURES_PHRASE)
    assert len(train["y"]) == len(train["idx_texte"]) == 1200
    meta = charger_split(tmp_path, "train")["y"]
    assert (train["y"] == meta[train["idx_texte"]]).all()  # les phrases héritent du label

    val = charger_phrases(tmp_path, "val")                 # toutes les phrases (5 par paragraphe)
    assert len(val["y"]) == 100 * 5

    assert 250 <= len(charger_phrases(tmp_path, "train", max_exemples=600)["y"]) <= 950


def test_reprise_apres_interruption(tmp_path, dataset_synthetique):
    construire_cache(dataset_synthetique, tmp_path, taille_chunk=100, encodeur=faux_encodeur, calculateur_ppl=faux_ppl)
    (tmp_path / "train_00002.npz").unlink()                  # simule un bloc manquant
    with pytest.raises(ValueError, match="incomplet"):
        charger_split(tmp_path, "train")

    compteur = Compteur()
    construire_cache(dataset_synthetique, tmp_path, taille_chunk=100, encodeur=compteur, calculateur_ppl=faux_ppl)
    assert compteur.appels == 1                              # seul le bloc manquant est recalculé
    assert charger_split(tmp_path, "train")["emb"].shape[0] == 400


def test_parametres_differents_refuses(tmp_path, dataset_synthetique):
    construire_cache(dataset_synthetique, tmp_path, encodeur=faux_encodeur, calculateur_ppl=faux_ppl)
    with pytest.raises(ValueError, match="autres paramètres"):
        construire_cache(dataset_synthetique, tmp_path, max_tokens=64, encodeur=faux_encodeur, calculateur_ppl=faux_ppl)


def test_limite_par_split(tmp_path, dataset_synthetique):
    construire_cache(dataset_synthetique, tmp_path, max_par_split={"train": 100},
                     encodeur=faux_encodeur, calculateur_ppl=faux_ppl)
    assert 80 <= len(charger_split(tmp_path, "train")["y"]) <= 120
    assert len(charger_split(tmp_path, "val")["y"]) == 100


def _parquet_mixte(dataset_synthetique, chemin):
    """Le dataset synthétique en Parquet, avec 1 texte sur 7 en français (à filtrer)."""
    df = dataset_synthetique.copy()
    df["lang"] = ["fr" if i % 7 == 3 else "en" for i in range(len(df))]
    df.to_parquet(chemin)
    return df[df["lang"] == "en"]


def test_iterer_textes_par_lots(tmp_path, dataset_synthetique):
    from projet_ia.detection.donnees import iterer_textes
    attendu = _parquet_mixte(dataset_synthetique, tmp_path / "d.parquet")
    lots = list(iterer_textes(tmp_path / "d.parquet", "train", "en", taille_lot=50,
                              lignes_par_lecture=37))        # lots et lectures décalés
    assert all(len(l) == 50 for l in lots[:-1]) and 0 < len(lots[-1]) <= 50
    assert sum(lots, []) == attendu[attendu.split == "train"]["texte"].tolist()


def test_cache_en_flux_identique_au_cache_en_memoire(tmp_path, dataset_synthetique):
    import numpy as np
    from projet_ia.detection.cache import construire_cache_flux
    from projet_ia.detection.donnees import charger_dataset
    _parquet_mixte(dataset_synthetique, tmp_path / "d.parquet")
    construire_cache(charger_dataset(tmp_path / "d.parquet", langue="en"), tmp_path / "memoire",
                     taille_chunk=70, encodeur=faux_encodeur, calculateur_ppl=faux_ppl)
    construire_cache_flux(tmp_path / "d.parquet", tmp_path / "flux", taille_chunk=70,
                          encodeur=faux_encodeur, calculateur_ppl=faux_ppl)
    for split in ("train", "val", "test"):
        a, b = charger_split(tmp_path / "memoire", split), charger_split(tmp_path / "flux", split)
        assert len(a["y"]) < {"train": 400, "val": 100, "test": 100}[split]   # français filtré
        for cle in ("emb", "style", "coh", "ppl", "y"):
            assert np.array_equal(a[cle], b[cle])
        assert a["meta"].equals(b["meta"])
        pa, pb = charger_phrases(tmp_path / "memoire", split), charger_phrases(tmp_path / "flux", split)
        assert np.array_equal(pa["emb"], pb["emb"]) and np.array_equal(pa["idx_texte"], pb["idx_texte"])
