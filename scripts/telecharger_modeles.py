"""
Télécharge les poids des modèles publiés en Release GitHub (CDC §8 : pas de
modèles dans Git) et vérifie leur empreinte SHA-256.

    python scripts/telecharger_modeles.py

La liste des modèles est dans scripts/modeles.json. Pour publier une nouvelle
version d'un modèle : créer une Release (ex. modele-detection-fr-v1), y joindre
le fichier, puis ajouter/mettre à jour l'entrée correspondante (url + sha256).

Sans dépendance (bibliothèque standard uniquement). Code de sortie 1 si un
téléchargement échoue ou si une empreinte ne correspond pas.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
MANIFESTE = Path(__file__).resolve().parent / "modeles.json"


def sha256(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def telecharger(entree: dict) -> bool:
    cible = RACINE / entree["fichier"]
    if cible.exists() and sha256(cible) == entree["sha256"]:
        print(f"[ok]   {entree['fichier']} (déjà présent)")
        return True

    cible.parent.mkdir(parents=True, exist_ok=True)
    temporaire = cible.with_suffix(cible.suffix + ".part")
    print(f"[...]  {entree['fichier']} <- {entree['url']}")
    try:
        with urllib.request.urlopen(entree["url"], timeout=120) as rep, open(temporaire, "wb") as out:
            while bloc := rep.read(1 << 20):
                out.write(bloc)
    except Exception as e:
        temporaire.unlink(missing_ok=True)
        print(f"[ERR]  téléchargement impossible : {e}", file=sys.stderr)
        return False

    obtenu = sha256(temporaire)
    if obtenu != entree["sha256"]:
        temporaire.unlink(missing_ok=True)
        print(f"[ERR]  empreinte incorrecte pour {entree['fichier']} : {obtenu}", file=sys.stderr)
        return False
    temporaire.replace(cible)
    print(f"[ok]   {entree['fichier']}")
    return True


def main() -> int:
    entrees = json.loads(MANIFESTE.read_text(encoding="utf-8"))
    resultats = [telecharger(e) for e in entrees]
    return 0 if all(resultats) else 1


if __name__ == "__main__":
    sys.exit(main())
