"""Petits utilitaires de gestion de fichiers."""

from pathlib import Path


def creer_dossier_parent(chemin) -> None:
    """Crée le dossier qui contiendra `chemin` s'il n'existe pas encore."""
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
