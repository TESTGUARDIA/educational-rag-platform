"""
Vérification automatique de réponses d'élèves.

    from projet_ia.verification import evaluer_reponse
    r = evaluer_reponse(
        question   = "Quelle est la capitale de la France ?",
        reponse    = "C'est Paris.",
        correction = "Paris est la capitale de la France.",
    )
    print(r["score"], r["verdict"])
"""

from .evaluateur import evaluer_reponse
from .poids import charger_poids_calibres, score_calibre

__all__ = ["evaluer_reponse", "charger_poids_calibres", "score_calibre"]
