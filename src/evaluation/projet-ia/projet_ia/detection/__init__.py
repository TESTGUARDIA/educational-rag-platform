"""
Détection de textes écrits par une IA ou par un humain, en français ou en anglais.

Un paragraphe est analysé de trois façons (paragraphe entier, phrase par phrase,
comparaison entre phrases), chacune par un MLP optimisé par algorithme génétique ;
leurs sorties sont fusionnées par une somme pondérée de log-odds apprise par
descente de gradient.

    from projet_ia.detection import (charger_dataset, construire_cache, algorithme_genetique,
                                     entrainer_final, predire_depuis_fichier)

    df = charger_dataset("data/dataset.parquet", langue="en")
    construire_cache(df, langue="en")                      # calcul unique des embeddings
    for c in ("paragraphe", "phrase", "interphrases"):
        algorithme_genetique("data/cache/en", composant=c)  # meilleur MLP de chaque composant
    entrainer_final(dossier_cache="data/cache/en")          # entraînement + fusion + test + rapport
    predire_depuis_fichier("Un paragraphe à analyser. Avec plusieurs phrases.")   # langue auto
"""

from .cache import construire_cache
from .donnees import charger_dataset
from .composants import COMPOSANTS
from .entrainement import entrainer_final
from .evolution import algorithme_genetique
from .mlp import GenomeMLP
from .prediction import predire_depuis_fichier, predire_textes

__all__ = ["charger_dataset", "construire_cache", "algorithme_genetique", "entrainer_final",
           "COMPOSANTS",
           "GenomeMLP", "predire_depuis_fichier", "predire_textes"]
