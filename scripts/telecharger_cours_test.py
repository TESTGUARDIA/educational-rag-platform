"""
Télécharge le jeu de cours de test (licences libres) dans cours_test/ (ignoré par Git).

    python scripts/telecharger_cours_test.py

La liste, avec la source et la licence de chaque document, est dans
scripts/cours_test.json. Les fichiers ne sont pas commités : le dépôt est public
et certaines licences (CC BY-NC-SA) imposent des conditions de redistribution.
Un fichier déjà présent n'est pas retéléchargé.
"""

import json
import sys
import urllib.request
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
DOSSIER = RACINE / "cours_test"
MANIFESTE = Path(__file__).resolve().parent / "cours_test.json"
ENTETES = {"User-Agent": "TestGuardAI-cours-test/1.0 (projet etudiant, usage pedagogique)"}


def telecharger(entree: dict) -> bool:
    cible = DOSSIER / entree["fichier"]
    if cible.exists() and cible.stat().st_size > 0:
        print(f"[ok]   {entree['fichier']} (déjà présent)")
        return True
    cible.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(entree["url"], headers=ENTETES)
        with urllib.request.urlopen(req, timeout=180) as rep:
            contenu = rep.read()
    except Exception as e:
        print(f"[ERR]  {entree['fichier']} : {e}", file=sys.stderr)
        return False
    if not contenu.startswith(b"%PDF"):
        print(f"[ERR]  {entree['fichier']} : la réponse n'est pas un PDF", file=sys.stderr)
        return False
    cible.write_bytes(contenu)
    print(f"[ok]   {entree['fichier']} ({len(contenu) // 1024} Ko, {entree['licence']})")
    return True


def main() -> int:
    entrees = json.loads(MANIFESTE.read_text(encoding="utf-8"))
    resultats = [telecharger(e) for e in entrees]
    (DOSSIER / "LISEZMOI.txt").write_text(
        "Cours de test téléchargés par scripts/telecharger_cours_test.py.\n"
        "NE PAS COMMITER. Sources et licences : scripts/cours_test.json\n", encoding="utf-8")
    print(f"\n{sum(resultats)}/{len(resultats)} documents dans {DOSSIER}")
    return 0 if all(resultats) else 1


if __name__ == "__main__":
    sys.exit(main())
