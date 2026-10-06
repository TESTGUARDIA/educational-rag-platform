import numpy as np

from projet_ia.detection.style import (
    N_FEATURES_COHERENCE, N_FEATURES_STYLE, NOMS_FEATURES_STYLE, calculer_coherence,
    calculer_style, decouper_phrases, features_style)

REGULIER = " ".join(["Le chat dort sur le canapé du salon."] * 5)
IRREGULIER = ("Non. Hier soir, après une journée interminable passée à courir entre les "
              "réunions et les appels, je suis enfin rentré chez moi. Bof ! Et toi ?")


def _f(texte):
    return dict(zip(NOMS_FEATURES_STYLE, features_style(texte)))


def test_noms_et_taille_coherents():
    assert len(NOMS_FEATURES_STYLE) == N_FEATURES_STYLE == len(features_style(REGULIER))


def test_texte_regulier_moins_variable_que_texte_humain():
    reg, irr = _f(REGULIER), _f(IRREGULIER)
    assert reg["phrase_lg_cv"] < irr["phrase_lg_cv"]
    assert reg["phrase_lg_std"] == 0.0
    assert reg["ratio_debuts_repetes"] > irr["ratio_debuts_repetes"]
    assert reg["jaccard_successives_moy"] == 1.0


def test_cas_limites_sans_nan():
    for texte in ["", "   ", "...", "!!! ???", "Un seul mot", "Une phrase.", "\n\n"]:
        v = features_style(texte)
        assert v.shape == (N_FEATURES_STYLE,) and np.isfinite(v).all(), texte


def test_decoupage_en_phrases():
    assert len(decouper_phrases("Un. Deux ! Trois ? Quatre…")) == 4
    assert decouper_phrases("vers un\nvers deux\nvers trois") == ["vers un", "vers deux", "vers trois"]


def test_calcul_parallele_identique_au_sequentiel():
    textes = [REGULIER, IRREGULIER] * 30
    assert np.array_equal(calculer_style(textes), calculer_style(textes, n_jobs=2, taille_lot=10))


def test_coherence_phrases_identiques_vs_differentes():
    def encodeur(phrases):                      # un vecteur différent par phrase distincte
        rng = lambda s: np.random.default_rng(abs(hash(s)) % 2**32).normal(size=32)
        return np.stack([rng(p) for p in phrases])

    c = calculer_coherence([REGULIER, IRREGULIER], encodeur)
    assert c.shape == (2, N_FEATURES_COHERENCE)
    assert c[0, 0] > 0.99                       # 5 phrases identiques -> similarité 1
    assert c[1, 0] < c[0, 0]


def test_coherence_texte_a_une_phrase():
    c = calculer_coherence(["Une seule phrase."], lambda p: np.ones((len(p), 4)))
    assert c.tolist() == [[1.0, 0.0, 1.0, 0.0]]
