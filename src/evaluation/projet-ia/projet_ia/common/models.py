"""
common/models.py
================
Chargement PARESSEUX des modèles lourds (spaCy, CamemBERT, NLI).

Avant, les modèles étaient chargés dès l'`import` des fichiers, ce qui rendait
chaque import très lent et impossible à tester sans les modèles installés.
Maintenant, un modèle n'est chargé qu'à sa première utilisation, puis gardé en
mémoire (cache `lru_cache`) : les appels suivants sont instantanés et tous les
modules partagent la même instance.
"""

from functools import lru_cache

from .. import config


@lru_cache(maxsize=None)
def get_spacy(nom: str):
    """Charge (une seule fois) un pipeline spaCy, ex. 'fr_core_news_md'."""
    import spacy

    print(f"Chargement de spaCy ({nom})...")
    return spacy.load(nom)


@lru_cache(maxsize=None)
def get_camembert(nom: str = config.MODELE_CAMEMBERT):
    """Charge (une seule fois) CamemBERT. Retourne (tokenizer, modele)."""
    from transformers import AutoTokenizer, AutoModel

    print(f"Chargement de CamemBERT ({nom})...")
    tokenizer = AutoTokenizer.from_pretrained(nom)
    modele = AutoModel.from_pretrained(nom)
    modele.eval()
    return tokenizer, modele


@lru_cache(maxsize=None)
def get_nli(modeles: tuple = config.MODELES_NLI_DETECTION):
    """
    Charge (une seule fois) un classifieur NLI zero-shot : le premier modèle
    de `modeles` qui se charge correctement.
    """
    from transformers import pipeline

    for modele in modeles:
        try:
            print(f"Chargement du classifieur NLI : {modele} ...")
            clf = pipeline("zero-shot-classification", model=modele)
            print(f"Classifieur NLI chargé : {modele}")
            return clf
        except Exception as e:
            print(f"Modèle indisponible ({modele}) : {e}")
    raise RuntimeError("Aucun modèle NLI n'a pu être chargé.")


@lru_cache(maxsize=None)
def get_encodeur(nom: str, appareil: str | None = None):
    """
    Charge (une seule fois) un encodeur Transformers pour la détection IA,
    placé sur GPU s'il y en a un. Retourne (tokenizer, modele, appareil_torch).
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    appareil = torch.device(appareil or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Chargement de l'encodeur {nom} sur {appareil}...")
    tokenizer = AutoTokenizer.from_pretrained(nom)
    modele = AutoModel.from_pretrained(nom).to(appareil)
    modele.eval()
    return tokenizer, modele, appareil


@lru_cache(maxsize=None)
def get_modele_causal(nom: str, appareil: str | None = None):
    """
    Charge (une seule fois) un modèle de langue causal (GPT-2, BLOOM…) pour les
    features de perplexité, placé sur GPU s'il y en a un. Il n'est jamais entraîné.
    Retourne (tokenizer, modele, appareil_torch).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    appareil = torch.device(appareil or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Chargement du modèle de langue {nom} sur {appareil}...")
    tokenizer = AutoTokenizer.from_pretrained(nom)
    modele = AutoModelForCausalLM.from_pretrained(nom).to(appareil)
    modele.eval()
    return tokenizer, modele, appareil
