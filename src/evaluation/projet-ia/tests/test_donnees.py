import pandas as pd
import pytest

from projet_ia.detection.donnees import charger_dataset, echantillon_stratifie


def _brut():
    return pd.DataFrame({
        "text": ["a b c.", "d e f.", "g h i.", "j k l."],
        "label": ["ai", "human", "ai", "human"],
        "split": ["train", "val", "test", "train"],
        "domain": ["news", "reddit", "news", "wiki"],
        "model": ["gpt-4", "None", "gpt-3", "None"],
    })


def test_lecture_parquet_et_jsonl(tmp_path):
    brut = _brut()
    brut.to_parquet(tmp_path / "d.parquet")
    brut.to_json(tmp_path / "d.jsonl", orient="records", lines=True)
    for nom in ("d.parquet", "d.jsonl"):
        df = charger_dataset(tmp_path / nom)
        assert df["label"].tolist() == [1, 0, 1, 0]
        assert df["split"].tolist() == ["train", "val", "test", "train"]
        assert df["domaine"].tolist() == ["news", "reddit", "news", "wiki"]
        assert df["source"].eq("inconnu").all()          # colonne optionnelle absente


def test_colonne_obligatoire_manquante(tmp_path):
    _brut().drop(columns=["split"]).to_parquet(tmp_path / "d.parquet")
    with pytest.raises(ValueError, match="split"):
        charger_dataset(tmp_path / "d.parquet")


def test_label_inconnu(tmp_path):
    brut = _brut(); brut["label"] = ["ai", "humain??", "ai", "human"]
    brut.to_parquet(tmp_path / "d.parquet")
    with pytest.raises(ValueError, match="label"):
        charger_dataset(tmp_path / "d.parquet")


def test_fichier_absent(tmp_path):
    with pytest.raises(FileNotFoundError):
        charger_dataset(tmp_path / "nexiste_pas.parquet")


def test_echantillon_stratifie(dataset_synthetique):
    sous = echantillon_stratifie(dataset_synthetique, 200, seed=1)
    assert 180 <= len(sous) <= 220
    assert abs(sous["label"].mean() - 0.5) < 0.05
    assert len(echantillon_stratifie(dataset_synthetique, None)) == len(dataset_synthetique)


def test_filtre_par_langue(tmp_path):
    brut = _brut(); brut["lang"] = ["en", "fr", "English", "french"]
    brut.to_parquet(tmp_path / "d.parquet")
    assert charger_dataset(tmp_path / "d.parquet", langue="en")["texte"].tolist() == ["a b c.", "g h i."]
    assert len(charger_dataset(tmp_path / "d.parquet", langue="fr")) == 2
    assert len(charger_dataset(tmp_path / "d.parquet")) == 4          # sans filtre


def test_sans_colonne_de_langue_on_garde_tout(tmp_path):
    _brut().to_parquet(tmp_path / "d.parquet")
    assert len(charger_dataset(tmp_path / "d.parquet", langue="fr")) == 4
