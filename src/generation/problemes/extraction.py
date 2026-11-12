"""
Extraction d'un cours PDF en Markdown + LaTeX, page par page, par un modèle de vision.

Pourquoi la vision : dans les cours rédigés sous Word (police Cambria Math), la couche
texte du PDF est inutilisable pour les formules. Sur le cours d'intégration testé,
« I_n » sort « 𝐼' », « x² » sort « 𝑥! » et le signe intégrale sort « Y ». On rend
donc chaque page en image et on la fait transcrire.

Chaque page transcrite est mise en cache (cache/<empreinte du PDF>/<modèle>_<dpi>/page_NN.md) :
l'extraction, lente, n'est faite qu'une fois par cours.
"""

import hashlib
import logging
from pathlib import Path

from .ollama import discuter

log = logging.getLogger("testguardai.problemes")

CONSIGNE_PAGE = """Transcris fidèlement cette page d'un cours de mathématiques en Markdown.

Règles :
- Toutes les formules en LaTeX : $...$ en ligne, $$...$$ en bloc (intégrales avec leurs bornes,
  indices, exposants, fractions). N'invente rien, ne résous rien, ne commente pas.
- Conserve les intitulés tels quels en gras : **Définition :**, **Propriété :**, **Théorème :**,
  **Méthode :**, **Démonstration :**, **Exemple :**, **Remarque :**, **Correction**.
- Titres de parties en Markdown : « # Partie 1 : ... », sous-titres « ## 1) ... ».
- Ignore l'en-tête et le pied de page (auteur, site web, numéro de page), les liens « Vidéo »,
  les mentions de copyright et le contenu des figures (écris seulement [figure : courte description]).
- Les tableaux en tableau Markdown.
Réponds uniquement avec la transcription."""


def empreinte(pdf: Path) -> str:
    return hashlib.sha256(pdf.read_bytes()).hexdigest()[:16]


def rendre_pages(pdf: Path, dpi: int = 120, pages: list[int] | None = None) -> list[tuple[int, bytes]]:
    """Images PNG des pages (numérotées à partir de 1)."""
    import pymupdf

    doc = pymupdf.open(pdf)
    numeros = pages or list(range(1, len(doc) + 1))
    return [(n, doc[n - 1].get_pixmap(dpi=dpi).tobytes("png")) for n in numeros if 1 <= n <= len(doc)]


def texte_brut(pdf: Path, pages: list[int] | None = None) -> str:
    """Couche texte du PDF (rapide, mais formules souvent illisibles) : repli et comparaison."""
    import pymupdf

    doc = pymupdf.open(pdf)
    numeros = pages or range(1, len(doc) + 1)
    return "\n\n".join(doc[n - 1].get_text() for n in numeros if 1 <= n <= len(doc))


def extraire_cours(pdf: Path, modele_vision: str, dossier_cache: Path, dpi: int = 120,
                   pages: list[int] | None = None) -> str:
    """Transcrit le PDF en Markdown + LaTeX (avec cache par page) et retourne le texte complet."""
    pdf = Path(pdf)
    variante = "".join(c if c.isalnum() else "_" for c in f"{modele_vision}_{dpi}dpi")
    cache = Path(dossier_cache) / empreinte(pdf) / variante
    cache.mkdir(parents=True, exist_ok=True)
    morceaux = []
    for numero, image in rendre_pages(pdf, dpi, pages):
        fichier = cache / f"page_{numero:02d}.md"
        if fichier.exists():
            morceaux.append(fichier.read_text(encoding="utf-8"))
            continue
        log.warning("Transcription de la page %d avec %s…", numero, modele_vision)
        r = discuter(modele_vision, [{"role": "user", "content": CONSIGNE_PAGE}],
                     images=[image], temperature=0.0, num_ctx=8192)
        texte = r["contenu"].strip()
        log.warning("  page %d : %d caractères en %.0f s", numero, len(texte), r["duree_s"])
        fichier.write_text(texte, encoding="utf-8")
        morceaux.append(texte)
    cours = "\n\n".join(morceaux)
    (cache / "cours.md").write_text(cours, encoding="utf-8")
    return cours
