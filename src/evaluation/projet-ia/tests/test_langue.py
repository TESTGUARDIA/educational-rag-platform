import pytest

from projet_ia import config
from projet_ia.detection.langue import detecter_langue


def test_detection_anglais_et_francais():
    assert detecter_langue("The cat was sitting on the mat and it was not happy with this.") == "en"
    assert detecter_langue("Le chat dormait sur le canapé et il ne voulait pas se lever.") == "fr"


def test_texte_trop_court_ou_ambigu():
    assert detecter_langue("ok") is None
    assert detecter_langue("Paris Berlin Madrid Rome Lisbon Vienna") is None


def test_chemins_par_langue():
    en, fr = config.CheminsDetection("en"), config.CheminsDetection("fr")
    assert en.cache != fr.cache and en.modele != fr.modele
    assert en.genome("phrase").name == "genome_phrase.json"
    with pytest.raises(ValueError):
        config.CheminsDetection("klingon")
    assert config.LANGUES["en"]["encodeur"] != config.LANGUES["fr"]["encodeur"]
