"""
Calibration des poids de la vérification (régression logistique par descente
de gradient).

    from projet_ia.calibration import calibrer
    calibrer("data/meta_dataset.csv")      # écrit models/poids_calibres.json
"""

from .pipeline import calibrer

__all__ = ["calibrer"]
