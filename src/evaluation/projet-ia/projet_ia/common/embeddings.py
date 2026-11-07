"""Embeddings CamemBERT partagés par la vérification et la détection."""

import contextlib

import numpy as np

from .models import get_camembert, get_encodeur


def embedding_cls(texte: str) -> np.ndarray:
    """Embedding CamemBERT d'un texte : vecteur du token [CLS] (768 dimensions)."""
    import torch

    tokenizer, modele = get_camembert()
    inputs = tokenizer(texte, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = modele(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


def encoder_textes(textes, nom_modele: str, batch_size: int = 64, max_length: int = 512,
                   pooling: str = "mean", appareil: str | None = None) -> np.ndarray:
    """
    Encode une liste de textes PAR LOTS (rapide, GPU si disponible).

    Les textes sont triés par longueur avant d'être regroupés (moins de padding,
    donc nettement plus rapide), puis remis dans l'ordre d'origine.

    Args:
        pooling : "mean" (moyenne des tokens, recommandé) ou "cls".
    Returns:
        tableau float32 de forme (len(textes), dimension_du_modele)
    """
    import torch

    textes = list(textes)
    if not textes:
        return np.zeros((0, 0), dtype=np.float32)

    tokenizer, modele, dev = get_encodeur(nom_modele, appareil)
    ordre = np.argsort([len(t) for t in textes])
    sortie = None

    with torch.no_grad():
        for debut in range(0, len(textes), batch_size):
            idx = ordre[debut:debut + batch_size]
            enc = tokenizer([textes[i] for i in idx], return_tensors="pt",
                            truncation=True, max_length=max_length, padding=True).to(dev)
            ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if dev.type == "cuda" else contextlib.nullcontext())
            with ctx:
                h = modele(**enc).last_hidden_state
            if pooling == "cls":
                v = h[:, 0, :]
            else:
                masque = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * masque).sum(dim=1) / masque.sum(dim=1).clamp(min=1)
            v = v.float().cpu().numpy()
            if sortie is None:
                sortie = np.empty((len(textes), v.shape[1]), dtype=np.float32)
            sortie[idx] = v
    return sortie
