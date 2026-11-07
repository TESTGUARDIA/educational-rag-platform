import numpy as np
import pytest

from conftest import faux_encodeur, faux_ppl
from projet_ia.detection.cache import charger_phrases, charger_split, construire_cache
from projet_ia.detection.perplexite import (N_FEATURES_PPL, N_FEATURES_PPL_PHRASE,
                                            bornes_phrases, features_perplexite)


def test_bornes_phrases():
    texte = "Une phrase. Puis une autre !  Et la fin."
    assert bornes_phrases(texte, ["Une phrase.", "Puis une autre !", "Et la fin."]).tolist() == \
        [0, 12, 30]


def test_tokens_repartis_entre_les_phrases():
    texte = "Aa bb. Cc dd."
    tokens = {"debut": np.array([0, 3, 7, 10]), "logprob": np.array([-1.0, -3.0, -5.0, -7.0]),
              "top1": np.array([1, 0, 0, 0]), "entropie": np.array([2.0, 2.0, 4.0, 4.0])}
    para, phrases = features_perplexite(texte, ["Aa bb.", "Cc dd."], tokens)
    assert para.shape == (N_FEATURES_PPL,) and phrases.shape == (2, N_FEATURES_PPL_PHRASE)
    assert phrases[:, 0].tolist() == [-2.0, -6.0]                  # log-prob moyenne par phrase
    assert phrases[:, 2].tolist() == [0.5, 0.0]                    # part de top-1
    assert para[0] == -4.0 and para[3] == 3.0                      # moyennes sur tout le texte
    assert para[4] == pytest.approx(4.0 / 3.0)                     # surprise / entropie
    assert para[5] == pytest.approx(2.0)                           # écart entre phrases


def test_phrase_tronquee_et_texte_vide():
    tokens = {"debut": np.array([0, 3]), "logprob": np.array([-1.0, -3.0]),
              "top1": np.array([0, 1]), "entropie": np.array([1.0, 1.0])}
    para, phrases = features_perplexite("Aa bb. Cc dd.", ["Aa bb.", "Cc dd."], tokens)
    assert np.allclose(phrases[1], para[:N_FEATURES_PPL_PHRASE])   # sans token : valeurs du texte
    assert para[5] == 0.0                                          # une seule phrase mesurée
    vide = {"debut": np.array([], dtype=int), "logprob": np.array([]), "top1": np.array([]),
            "entropie": np.array([])}
    para, phrases = features_perplexite("", [""], vide)
    assert not para.any() and not phrases.any()


def test_cache_contient_la_perplexite(tmp_path, dataset_synthetique):
    construire_cache(dataset_synthetique, tmp_path, taille_chunk=150, encodeur=faux_encodeur,
                     calculateur_ppl=faux_ppl)
    train = charger_split(tmp_path, "train")
    assert train["ppl"].shape == (400, N_FEATURES_PPL)
    ia = train["y"] == 1                                           # textes « robot » plus prévisibles
    assert train["ppl"][ia, 0].mean() > train["ppl"][~ia, 0].mean()
    phrases = charger_phrases(tmp_path, "val")
    assert phrases["ppl"].shape == (100 * 5, N_FEATURES_PPL_PHRASE)


@pytest.mark.modeles
def test_vrai_modele_de_langue():
    from projet_ia.detection.perplexite import calculer_tokens

    previsible = "One, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve."
    bizarre = "Purple elbows negotiate quietly beneath sarcastic volcanoes of yesterday's soup."
    sortie = calculer_tokens([previsible, bizarre, ""], "gpt2")
    for s in sortie:
        assert len(s["debut"]) == len(s["logprob"]) == len(s["top1"]) == len(s["entropie"])
        assert (s["logprob"] <= 0).all() and (s["entropie"] >= -1e-4).all()
    assert len(sortie[2]["logprob"]) == 0
    assert sortie[0]["logprob"].mean() > sortie[1]["logprob"].mean() + 1.0
