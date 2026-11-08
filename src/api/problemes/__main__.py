"""
Prototype de génération de sujets de mathématiques (pôle Génération).

    python -m src.api.problemes extraire  cours.pdf --modele-vision qwen2.5vl:7b
    python -m src.api.problemes analyser  sortie_problemes/cours.md --modele qwen3:8b
    python -m src.api.problemes probleme  sortie_problemes/analyse.json --theme "intégrales de Wallis" --parties 5,6
    python -m src.api.problemes questions sortie_problemes/analyse.json --nb 5
    python -m src.api.problemes tout      cours.pdf --theme "intégrales de Wallis" --parties 5,6

Les résultats vont dans sortie_problemes/ (ignoré par Git : les cours peuvent être protégés).
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

SORTIE = Path("sortie_problemes")


def _pages(texte: str | None) -> list[int] | None:
    if not texte:
        return None
    pages = []
    for morceau in texte.split(","):
        a, _, b = morceau.partition("-")
        pages += list(range(int(a), int(b or a) + 1))
    return pages


def _raisonnement(v: str):
    return {"non": False, "oui": True}.get(v, v)    # "low" | "medium" | "high" pour gpt-oss


def _ecrire(nom: str, contenu) -> Path:
    SORTIE.mkdir(exist_ok=True)
    chemin = SORTIE / nom
    if isinstance(contenu, str):
        chemin.write_text(contenu, encoding="utf-8")
    else:
        chemin.write_text(json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"→ {chemin}")
    return chemin


def cmd_extraire(a) -> Path:
    from .extraction import extraire_cours, texte_brut
    if a.texte:
        cours = texte_brut(Path(a.pdf))
    else:
        cours = extraire_cours(Path(a.pdf), a.modele_vision, SORTIE / "cache", a.dpi, _pages(a.pages))
    return _ecrire("cours.md", cours)


def cmd_analyser(a) -> Path:
    from .analyse import analyser_cours
    analyse = analyser_cours(Path(a.cours).read_text(encoding="utf-8"), a.modele, a.num_ctx)
    for p in analyse["parties"]:
        print(f"  Partie {p['numero']} « {p['titre']} » : {len(p['notions'])} notions "
              f"({p['blocs_reperes']} blocs repérés, {p['duree_s']:.0f} s)"
              + (f"  ⚠ ÉCHEC : {p['erreur']}" if p.get("erreur") else ""))
    return _ecrire("analyse.json", analyse)


def cmd_probleme(a) -> Path:
    from .analyse import fiches_pour
    from .generation import generer_probleme, rendre_markdown
    analyse = json.loads(Path(a.analyse).read_text(encoding="utf-8"))
    parties = [int(x) for x in a.parties.split(",")] if a.parties else None
    fiches = fiches_pour(analyse, parties, a.importance_min)
    print(f"  {len(fiches)} notions retenues")
    res = generer_probleme(fiches, a.theme, a.modele, raisonnement=_raisonnement(a.raisonnement),
                           difficulte=a.difficulte, niveau=a.niveau, tours_max=a.tours,
                           num_ctx=a.num_ctx)
    slug = re.sub(r"[^a-z0-9]+", "_", f"{a.theme}_{a.modele}".lower()).strip("_")
    _ecrire(f"probleme_{slug}.json", res)
    return _ecrire(f"probleme_{slug}.md", rendre_markdown(res))


def cmd_questions(a) -> Path:
    from .analyse import fiches_pour
    from .generation import generer_questions_cours
    analyse = json.loads(Path(a.analyse).read_text(encoding="utf-8"))
    parties = [int(x) for x in a.parties.split(",")] if a.parties else None
    res = generer_questions_cours(fiches_pour(analyse, parties, 3), a.nb, a.modele,
                                  niveau=a.niveau, num_ctx=a.num_ctx)
    md = "\n\n".join(f"**{i}. [{q['type_question']} — {q['notion']}]** {q['question']}\n\n"
                     f"> {q['reponse']}" for i, q in enumerate(res["questions"], 1))
    _ecrire("questions_cours.json", res)
    return _ecrire("questions_cours.md", f"# Questions de cours ({res['duree_s']:.0f} s, "
                                         f"{res['modele']})\n\n{md}")


def main(argv=None):
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="python -m src.api.problemes", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sous = ap.add_subparsers(dest="commande", required=True)

    def communs(p, modele=True):
        if modele:
            p.add_argument("--modele", default="qwen3:8b", help="modèle Ollama de génération")
        p.add_argument("--num-ctx", type=int, default=16384,
                       help="fenêtre de contexte (réduire si la mémoire vidéo sature)")
        p.add_argument("--niveau", default="Terminale, spécialité mathématiques")

    def options_extraction(p):
        p.add_argument("--modele-vision", default="qwen2.5vl:7b")
        p.add_argument("--dpi", type=int, default=120)
        p.add_argument("--pages", help="ex. 1-5,8 (toutes par défaut)")
        p.add_argument("--texte", action="store_true",
                       help="couche texte du PDF au lieu de la vision (rapide, formules abîmées)")

    def options_probleme(p):
        p.add_argument("--theme", required=True)
        p.add_argument("--parties", help="numéros de parties du cours, ex. 5,6")
        p.add_argument("--difficulte", type=int, default=2, choices=[1, 2, 3])
        p.add_argument("--raisonnement", default="oui", help="oui | non | low | medium | high")
        p.add_argument("--tours", type=int, default=2, help="tours de correction max")
        p.add_argument("--importance-min", type=int, default=2, choices=[1, 2, 3])

    p = sous.add_parser("extraire"); p.add_argument("pdf"); options_extraction(p)
    p.set_defaults(f=cmd_extraire)
    p = sous.add_parser("analyser"); p.add_argument("cours"); communs(p)
    p.set_defaults(f=cmd_analyser)
    p = sous.add_parser("probleme"); p.add_argument("analyse"); communs(p); options_probleme(p)
    p.set_defaults(f=cmd_probleme)
    p = sous.add_parser("questions"); p.add_argument("analyse"); communs(p)
    p.add_argument("--nb", type=int, default=5); p.add_argument("--parties")
    p.set_defaults(f=cmd_questions)
    p = sous.add_parser("tout"); p.add_argument("pdf"); communs(p); options_extraction(p)
    options_probleme(p)

    def cmd_tout(a):
        a.cours = str(cmd_extraire(a))
        a.analyse = str(cmd_analyser(a))
        cmd_probleme(a)
    p.set_defaults(f=cmd_tout)

    a = ap.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
