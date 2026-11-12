"""
Banc de mesure du pôle Génération.

    python -m src.generation.banc liste
    python -m src.generation.banc lancer                          # cas et configs par défaut
    python -m src.generation.banc lancer --cas maths_integrales_wallis --configs qwen3-8b-raisonnement,gpt-oss-20b
    python -m src.generation.banc resumer sortie_banc/2026-11-20_1430

Prérequis : python scripts/telecharger_cours_test.py (cours dans cours_test/) et les modèles
Ollama des configurations (voir configs.json).
"""

import argparse
import logging
import sys
from pathlib import Path

from . import banc


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m src.generation.banc", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = ap.add_subparsers(dest="commande", required=True)
    sous.add_parser("liste", help="cas et configurations disponibles")
    p = sous.add_parser("lancer", help="exécute le banc")
    p.add_argument("--cas", help="identifiants séparés par des virgules (tous par défaut)")
    p.add_argument("--configs", help="noms séparés par des virgules (celles marquées par_defaut sinon)")
    p.add_argument("--tours", type=int, default=2, help="tours de correction des problèmes")
    p = sous.add_parser("resumer", help="fusionne notation.csv et mesures -> synthese.md")
    p.add_argument("dossier")
    a = ap.parse_args(argv)

    cas = banc.charger_json("cas.json")
    configs = banc.charger_json("configs.json")
    if a.commande == "liste":
        print("Cas :")
        for c in cas:
            dispo = "ok" if banc.chemin_pdf(c) else "PDF absent"
            print(f"  {c['id']:32} {c['mode']:10} {c.get('matiere', ''):16} [{dispo}]")
        print("Configurations (* = par défaut) :")
        for c in configs:
            print(f"  {'*' if c.get('par_defaut') else ' '} {c['nom']:28} {c['modele']:14} "
                  f"raisonnement={c.get('raisonnement')} extraction={c.get('extraction')}")
    elif a.commande == "lancer":
        run = banc.lancer(banc.selectionner(cas, "id", a.cas),
                          banc.selectionner(configs, "nom", a.configs, lambda c: c.get("par_defaut")),
                          tours=a.tours)
        print(f"\nTerminé : {run}\n→ notez {run / 'a_noter.md'} dans {run / 'notation.csv'}"
              f" (grille : src/generation/banc/GRILLE.md)\n→ puis : python -m src.generation.banc resumer {run}")
    else:
        print(f"→ {banc.resumer(Path(a.dossier))}")


if __name__ == "__main__":
    main()
