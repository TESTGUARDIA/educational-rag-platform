"""Fixtures partagées : faux encodeur, fausse perplexité et dataset synthétique (aucun
modèle à télécharger)."""

import re
import zlib

try:
    import torch  # noqa: F401  (avant pandas : voir projet_ia/detection/__init__.py)
except ImportError:
    pass
import numpy as np
import pandas as pd
import pytest

DIM_FAUX = 64


def faux_encodeur(textes) -> np.ndarray:
    """Encodeur factice déterministe : un vecteur aléatoire par texte, décalé si le texte parle de 'robot'."""
    sortie = np.zeros((len(textes), DIM_FAUX), dtype=np.float32)
    for i, t in enumerate(textes):
        v = np.random.default_rng(zlib.crc32(t.encode("utf-8"))).normal(size=DIM_FAUX)
        if "robot" in t:
            v[:5] += 2.0
        sortie[i] = v
    return sortie


def faux_ppl(textes) -> list[dict]:
    """Perplexité factice déterministe : un « token » par mot, plus prévisible (log-prob plus
    haute) si le texte parle de 'robot'. Même format que perplexite.calculer_tokens."""
    sortie = []
    for t in textes:
        debuts = np.array([m.start() for m in re.finditer(r"\S+", t)], dtype=np.int32)
        rng = np.random.default_rng(zlib.crc32(t.encode("utf-8")))
        centre = -2.0 if "robot" in t else -4.0
        sortie.append({"debut": debuts, "logprob": rng.normal(centre, 1.0, len(debuts)),
                       "top1": rng.random(len(debuts)) < 0.4,
                       "entropie": rng.gamma(2.0, 1.5, len(debuts))})
    return sortie


def _paragraphe_ia(rng) -> str:
    """Phrases très régulières (12 mots chacune) + mot-clé 'robot'."""
    phrases = []
    for _ in range(5):
        mots = ["le", "robot"] + [f"mot{rng.integers(0, 30)}" for _ in range(10)]
        phrases.append(" ".join(mots).capitalize() + ".")
    return " ".join(phrases)


def _paragraphe_humain(rng) -> str:
    """Phrases de longueurs très variées."""
    phrases = []
    for _ in range(5):
        n = int(rng.integers(2, 26))
        mots = [f"mot{rng.integers(0, 30)}" for _ in range(n)]
        phrases.append(" ".join(mots).capitalize() + rng.choice([".", "!", "?"]))
    return " ".join(phrases)


@pytest.fixture
def dataset_synthetique() -> pd.DataFrame:
    """Dataset aux colonnes standard : 400 train / 100 val / 100 test, équilibré 1:1."""
    rng = np.random.default_rng(0)
    lignes = []
    for split, n in (("train", 400), ("val", 100), ("test", 100)):
        for i in range(n):
            ia = i % 2 == 0
            lignes.append({
                "texte": _paragraphe_ia(rng) if ia else _paragraphe_humain(rng),
                "label": int(ia), "split": split,
                "domaine": ["news", "reddit"][i % 4 < 2], "source": "synth",
                "generateur": "gpt-x" if ia else "None", "longueur": "medium",
            })
    return pd.DataFrame(lignes)
