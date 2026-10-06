"""Analyse de question et grammaire : nécessitent spaCy + les modèles français (sinon ignorés).

    pip install spacy
    python -m spacy download fr_core_news_md
    python -m spacy download fr_core_news_sm
"""

import pytest

spacy = pytest.importorskip("spacy")


def _modele_disponible(nom: str) -> bool:
    try:
        spacy.load(nom)
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _modele_disponible("fr_core_news_md"), reason="fr_core_news_md absent")
class TestQuestion:
    def test_type_et_caractere(self):
        from projet_ia.verification.question import analyser_question

        assert analyser_question("Qui a écrit Hamlet ?")["type_question"] == "personne"
        assert analyser_question("Combien font 15 × 8 ?")["type_question"] == "quantité"
        assert analyser_question("Quel est le rôle du cœur ?")["caractere_question"] == "semi-ouverte"
        assert analyser_question("Pourquoi le ciel est bleu ?")["caractere_question"] == "ouverte"

    def test_sous_questions(self):
        from projet_ia.verification.question import detecter_sous_questions

        assert len(detecter_sous_questions("Quel est le plus grand océan du monde et sa taille ?")) == 2
        assert len(detecter_sous_questions("Quelle est la capitale de la France ?")) == 1

    def test_coreference(self):
        from projet_ia.verification.question import resoudre_coreferences

        r = resoudre_coreferences("C'est Paris.", "Quelle est la capitale de la France ?")
        assert r.startswith("capitale est")


@pytest.mark.skipif(not _modele_disponible("fr_core_news_sm"), reason="fr_core_news_sm absent")
def test_statistiques_de_phrase():
    from projet_ia.detection.grammaire import stats_phrase

    assert stats_phrase("Elle se levait chaque matin, pleine d'espoir.") == {
        "total_mots": 8, "total_ponct": 2}
