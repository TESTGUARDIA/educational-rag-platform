"""
verification/poids.py
=====================
Poids calibrés (régression logistique) et calcul du score final calibré.

Les poids sont lus dans models/poids_calibres.json, généré par
`python -m projet_ia calibrer` (voir le paquet projet_ia.calibration).
Si ce fichier est absent, des poids de secours sont utilisés.
Ne pas modifier le JSON à la main : relancer la calibration à la place.
"""

import json
import math
import os
from functools import lru_cache

from .. import config


# Poids de secours, utilisés uniquement si poids_calibres.json est introuvable
# (permet au script de tourner même avant la première calibration).
POIDS_CALIBRES_DEFAUT = {
    "nli":                 +20.7409,
    "similarite_ponderee": +5.3117,
    "sens_global":         +8.9348,
    "penalite_neg":        -10.3482,
    "penalite_entites":    -6.0026,
    "penalite_concept":    +11.2731,
    "q_fermee":            +0.0000,
    "q_semi_ouverte":      +0.0000,
    "q_ouverte":           +0.0000,
    "biais":               -22.6737,
}


def charger_poids_calibres(chemin=config.CHEMIN_POIDS_CALIBRES) -> dict:
    """
    Charge les poids calibrés depuis poids_calibres.json. Si le fichier est
    absent ou invalide, retombe sur POIDS_CALIBRES_DEFAUT et avertit l'utilisateur.
    """
    chemin = str(chemin)
    if os.path.exists(chemin):
        try:
            with open(chemin, "r", encoding="utf-8") as f:
                data = json.load(f)
            poids = data["poids"]
            print(f"Poids calibrés chargés depuis {os.path.basename(chemin)} "
                  f"(entraîné sur {data.get('n_exemples_entrainement', '?')} exemples, "
                  f"F1 CV = {data.get('validation_croisee', {}).get('f1_moyen', '?')})")
            return poids
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{os.path.basename(chemin)} invalide ({e}) — utilisation des poids de secours.")
    else:
        print(f"{os.path.basename(chemin)} introuvable — utilisation des poids de secours.\n"
              f"  -> Lancez 'python -m projet_ia calibrer' pour générer des poids "
              f"calibrés sur vos propres données.")
    return dict(POIDS_CALIBRES_DEFAUT)


@lru_cache(maxsize=1)
def get_poids() -> dict:
    """Poids calibrés par défaut, chargés une seule fois (à la première utilisation)."""
    return charger_poids_calibres()


def score_calibre(scores: dict, poids: dict | None = None) -> float:
    """
    Calcule le score final avec les poids calibrés par descente de gradient
    (régression logistique, voir calibration_entrainement.py).

    Args:
        poids  : dict de poids à utiliser (par défaut : poids_calibres.json,
                 ou les poids de secours si le fichier est absent)
        scores : dict avec les clés nli, similarite_ponderee, sens_global,
                 penalite_neg, penalite_entites, penalite_concept,
                 caractere_question (ou type_question — la valeur doit
                 correspondre à une catégorie "q_<valeur>" vue à
                 l'entraînement ; sinon elle est simplement ignorée).

    Returns:
        Score entre 0 et 1 (sigmoid des poids × features)
    """
    poids = poids if poids is not None else get_poids()
    c = scores.get("caractere_question") or scores.get("type_question") or ""
    cle_categorie = f"q_{c}"

    z = (
        poids.get("nli", 0.0)                 * scores.get("nli", 0.0)
        + poids.get("similarite_ponderee", 0.0) * scores.get("similarite_ponderee", 0.0)
        + poids.get("sens_global", 0.0)         * scores.get("sens_global", 0.0)
        + poids.get("penalite_neg", 0.0)        * scores.get("penalite_neg", 0.0)
        + poids.get("penalite_entites", 0.0)    * scores.get("penalite_entites", 0.0)
        + poids.get("penalite_concept", 0.0)    * scores.get("penalite_concept", 0.0)
        + poids.get(cle_categorie, 0.0)
        + poids.get("biais", 0.0)
    )
    return 1.0 / (1.0 + math.exp(-z))  # sigmoid
