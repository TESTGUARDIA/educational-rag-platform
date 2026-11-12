"""
Banc de mesure du pôle Génération (phase 0 du plan).

Compare des configurations (modèle, raisonnement, méthode d'extraction) sur un jeu de cas
fixe (cas.json, cours de cours_test/), avec :
  - des mesures automatiques (durées, vérifications du corrigé, doublons, traçabilité) ;
  - une notation enseignant à l'aveugle (GRILLE.md) : les productions reçoivent un code
    (C001…) sans le nom du modèle ; la correspondance est gardée à part.

Arborescence d'un lancement : sortie_banc/<date>/
    <config>/<cas>/        sorties brutes (.md, .json) et metriques.json
    a_noter.md             toutes les productions à noter, anonymisées, groupées par cas
    notation.csv           à remplir (séparateur « ; »)
    correspondance.json    code -> configuration (à n'ouvrir qu'après la notation)
    metriques.json         toutes les mesures automatiques
"""

import csv
import datetime
import hashlib
import json
import logging
import os
import random
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean

from ..problemes import analyse as an
from ..problemes import extraction as ex
from ..problemes import generation as gen

log = logging.getLogger("testguardai.banc")

RACINE = Path(__file__).resolve().parents[3]
DOSSIER_BANC = Path(__file__).resolve().parent
SORTIE = RACINE / "sortie_banc"
CRITERES = ["exactitude", "conformite_programme", "interet", "difficulte_adaptee", "clarte"]
COLONNES_NOTATION = ["code", "cas", "type"] + CRITERES + ["utilisable", "commentaire"]


# ─── Chargement ────────────────────────────────────────────────────────────────
def charger_json(nom: str) -> list[dict]:
    return json.loads((DOSSIER_BANC / nom).read_text(encoding="utf-8"))


def selectionner(elements: list[dict], cle: str, noms: str | None, defaut=lambda e: True) -> list[dict]:
    if not noms:
        return [e for e in elements if defaut(e)]
    voulus = [n.strip() for n in noms.split(",")]
    inconnus = set(voulus) - {e[cle] for e in elements}
    if inconnus:
        raise ValueError(f"inconnu(s) : {sorted(inconnus)}")
    return [e for e in elements if e[cle] in voulus]


def chemin_pdf(cas: dict) -> Path | None:
    if cas.get("pdf"):
        p = RACINE / cas["pdf"]
    elif cas.get("pdf_env") and os.environ.get(cas["pdf_env"]):
        p = Path(os.environ[cas["pdf_env"]])
    else:
        return None
    return p if p.exists() else None


def _pages(texte: str | None) -> list[int] | None:
    if not texte:
        return None
    pages = []
    for morceau in str(texte).split(","):
        a, _, b = morceau.partition("-")
        pages += list(range(int(a), int(b or a) + 1))
    return pages


def _raisonnement(v):
    return {"non": False, "oui": True}.get(v, v)


def _slug(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


# ─── Mesures automatiques ──────────────────────────────────────────────────────
def metriques_extraction(texte: str) -> dict:
    return {"caracteres": len(texte),
            "formules_latex": len(re.findall(r"\$[^$]+\$", texte)),
            "caracteres_illisibles": texte.count("�")}


def quasi_doublons(textes: list[str], seuil: float = 0.85) -> int:
    """Nombre de paires de textes quasi identiques (ratio de similarité >= seuil)."""
    norm = [re.sub(r"\s+", " ", t.lower()).strip() for t in textes]
    return sum(SequenceMatcher(None, norm[i], norm[j]).ratio() >= seuil
               for i in range(len(norm)) for j in range(i + 1, len(norm)))


def metriques_analyse(analyse: dict) -> dict:
    notions = [n for p in analyse["parties"] for n in p["notions"]]
    return {"parties": len(analyse["parties"]), "notions": len(notions),
            "notions_importance_3": sum(n.get("importance") == 3 for n in notions),
            "parties_en_echec": sum(bool(p.get("erreur")) for p in analyse["parties"])}


def metriques_questions(res: dict) -> dict:
    qs = res["questions"]
    n = len(qs)
    return {"questions": n,
            "notion_trouvee_dans_le_cours_pct": round(100 * sum(q.get("notion_trouvee_dans_le_cours", False)
                                                              for q in qs) / n, 1) if n else 0.0,
            "quasi_doublons": quasi_doublons([q["question"] for q in qs]),
            "ecartees_question_egale_reponse": res.get("ecartees_question_egale_reponse", 0),
            "types": {t: sum(q.get("type_question") == t for q in qs)
                      for t in sorted({q.get("type_question") for q in qs})},
            "duree_generation_s": res.get("duree_s", 0.0),
            "duree_par_question_s": round(res.get("duree_s", 0.0) / n, 1) if n else None}


def metriques_probleme(res: dict) -> dict:
    v = res["verification"]
    comparaisons = res["resolution"]["comparaisons"]
    return {"questions": len(res["probleme"]["questions"]),
            "verifications_ok": v["ok"], "verifications_fausses": v["faux"],
            "verifications_illisibles": v["erreur"],
            "taux_verif_ok_pct": round(100 * v["ok"] / max(1, v["ok"] + v["faux"] + v["erreur"]), 1),
            "discordances_resolution": sum(c["statut"] == "discordant" for c in comparaisons),
            "erreurs_restantes": len(res["erreurs_restantes"]),
            "tours_correction": len(res["historique"]) - 1,
            "note_relecture_modele": res["critique"]["note"],
            "duree_generation_s": res["duree_totale_s"]}


# ─── Étapes avec cache ─────────────────────────────────────────────────────────
def extraire(cas: dict, config: dict, pdf: Path, cache: Path) -> tuple[str, str, float]:
    mode = cas.get("extraction") or config.get("extraction", "vision")
    t0 = time.monotonic()
    if mode == "texte":
        texte = ex.texte_brut(pdf, _pages(cas.get("pages")))
    else:
        texte = ex.extraire_cours(pdf, config["modele_vision"], cache / "extraction",
                                  config.get("dpi", 120), _pages(cas.get("pages")))
    return texte, mode, round(time.monotonic() - t0, 1)


def analyser(texte: str, cas: dict, config: dict, cache: Path) -> tuple[dict, float]:
    # La consigne fait partie de la clé : modifier le prompt invalide le cache
    cle = hashlib.sha256(f"{texte}|{cas.get('matiere')}|{an.CONSIGNE}".encode()).hexdigest()[:16]
    fichier = cache / "analyse" / f"{cle}_{_slug(config['modele'])}.json"
    if fichier.exists():
        return json.loads(fichier.read_text(encoding="utf-8")), 0.0
    t0 = time.monotonic()
    analyse = an.analyser_cours(texte, config["modele"], matiere=cas.get("matiere", "mathématiques"))
    fichier.parent.mkdir(parents=True, exist_ok=True)
    fichier.write_text(json.dumps(analyse, ensure_ascii=False, indent=1), encoding="utf-8")
    return analyse, round(time.monotonic() - t0, 1)


def _ecrire(chemin: Path, contenu) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(contenu if isinstance(contenu, str) else
                      json.dumps(contenu, ensure_ascii=False, indent=1), encoding="utf-8")


def executer_cas(cas: dict, config: dict, dossier: Path, cache: Path, tours: int) -> tuple[dict, list[dict]]:
    """Retourne (mesures, productions à noter). Une erreur est enregistrée, jamais propagée."""
    mesures = {"cas": cas["id"], "config": config["nom"], "mode": cas["mode"], "statut": "ok"}
    productions = []
    pdf = chemin_pdf(cas)
    if pdf is None:
        mesures["statut"] = "ignoré (PDF absent)"
        return mesures, productions
    try:
        texte, mode_extraction, d_ext = extraire(cas, config, pdf, cache)
        mesures.update(extraction=mode_extraction, duree_extraction_s=d_ext, **metriques_extraction(texte))
        analyse, d_an = analyser(texte, cas, config, cache)
        mesures.update(duree_analyse_s=d_an, **metriques_analyse(analyse))
        parties = [int(x) for x in str(cas["parties"]).split(",")] if cas.get("parties") else None

        if cas["mode"] == "questions":
            fiches = an.fiches_pour(analyse, parties, 3) or an.fiches_pour(analyse, parties, 2)
            res = gen.generer_questions_cours(fiches, cas.get("nb", 5), config["modele"],
                                              niveau=cas.get("niveau", "lycée"),
                                              matiere=cas.get("matiere", "mathématiques"))
            mesures.update(metriques_questions(res))
            _ecrire(dossier / "questions.json", res)
            for i, q in enumerate(res["questions"], 1):
                productions.append({"cas": cas["id"], "type": "question de cours", "config": config["nom"],
                                    "texte": f"**[{q['type_question']} — {q['notion']}]** {q['question']}"
                                             f"\n\n> Réponse attendue : {q['reponse']}",
                                    "source": str(dossier / "questions.json") + f"#{i}"})
        else:
            fiches = an.fiches_pour(analyse, parties, 2)
            res = gen.generer_probleme(fiches, cas["theme"], config["modele"],
                                       raisonnement=_raisonnement(config.get("raisonnement", "oui")),
                                       niveau=cas.get("niveau", "Terminale"), tours_max=tours)
            mesures.update(metriques_probleme(res))
            _ecrire(dossier / "probleme.json", res)
            md = gen.rendre_markdown(res)
            _ecrire(dossier / "probleme.md", md)
            sujet_et_corrige = md.split("\n---\n# Rapport de génération")[0]   # sans le rapport
            productions.append({"cas": cas["id"], "type": "problème", "config": config["nom"],
                                "texte": sujet_et_corrige, "source": str(dossier / "probleme.md")})
    except Exception as e:   # un cas qui échoue ne doit pas arrêter le banc
        log.exception("Échec %s / %s", config["nom"], cas["id"])
        mesures.update(statut="erreur", erreur=f"{type(e).__name__} : {e}")
    _ecrire(dossier / "metriques.json", mesures)
    return mesures, productions


# ─── Lancement et notation à l'aveugle ─────────────────────────────────────────
def lancer(cas_liste: list[dict], configs: list[dict], sortie: Path = SORTIE, tours: int = 2,
           graine: int = 0) -> Path:
    run = sortie / datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    run.mkdir(parents=True, exist_ok=True)
    toutes_mesures, productions = [], []
    for config in configs:
        for cas in cas_liste:
            log.warning("▶ %s / %s", config["nom"], cas["id"])
            m, p = executer_cas(cas, config, run / _slug(config["nom"]) / cas["id"], sortie / "cache", tours)
            toutes_mesures.append(m)
            productions += p
            _ecrire(run / "metriques.json", toutes_mesures)    # sauvegarde au fil de l'eau
    preparer_notation(run, productions, cas_liste, graine)
    return run


def preparer_notation(run: Path, productions: list[dict], cas_liste: list[dict], graine: int = 0) -> None:
    hasard = random.Random(graine)
    hasard.shuffle(productions)
    ordre_cas = {c["id"]: i for i, c in enumerate(cas_liste)}
    productions.sort(key=lambda p: ordre_cas.get(p["cas"], 99))   # groupées par cas, mélangées dedans
    correspondance, lignes_md = {}, ["# Productions à noter (anonymisées)\n",
                                     "Grille : `src/generation/banc/GRILLE.md`. Notez dans `notation.csv`.\n"]
    cas_par_id = {c["id"]: c for c in cas_liste}
    cas_courant = None
    with open(run / "notation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(COLONNES_NOTATION)
        for i, p in enumerate(productions, 1):
            code = f"C{i:03d}"
            correspondance[code] = {"config": p["config"], "cas": p["cas"], "source": p["source"]}
            if p["cas"] != cas_courant:
                cas_courant = p["cas"]
                c = cas_par_id.get(cas_courant, {})
                lignes_md.append(f"\n---\n## Cas « {cas_courant} » — {c.get('matiere', '')}, "
                                 f"{c.get('niveau', '')}" + (f", thème : {c['theme']}" if c.get("theme") else ""))
                if c.get("reference_humaine"):
                    lignes_md.append(f"Référence humaine pour comparer : {c['reference_humaine']}")
            lignes_md.append(f"\n### {code} ({p['type']})\n\n{p['texte']}\n")
            w.writerow([code, p["cas"], p["type"]] + [""] * (len(CRITERES) + 2))
    _ecrire(run / "a_noter.md", "\n".join(lignes_md))
    _ecrire(run / "correspondance.json", correspondance)


# ─── Synthèse ──────────────────────────────────────────────────────────────────
def _note(v: str) -> float | None:
    v = (v or "").strip().replace(",", ".")
    try:
        x = float(v)
        return x if 0 <= x <= 2 else None
    except ValueError:
        return None


def resumer(run: Path) -> Path:
    mesures = json.loads((run / "metriques.json").read_text(encoding="utf-8"))
    correspondance = json.loads((run / "correspondance.json").read_text(encoding="utf-8"))
    with open(run / "notation.csv", encoding="utf-8-sig") as f:
        notes = list(csv.DictReader(f, delimiter=";"))

    configs = list(dict.fromkeys(m["config"] for m in mesures))
    lignes = []
    for cfg in configs:
        ms = [m for m in mesures if m["config"] == cfg]
        ok = [m for m in ms if m["statut"] == "ok"]
        pb = [m for m in ok if m["mode"] == "probleme"]
        qc = [m for m in ok if m["mode"] == "questions"]
        notees = [n for n in notes if correspondance.get(n["code"], {}).get("config") == cfg]
        par_critere = {c: [x for x in (_note(n.get(c)) for n in notees) if x is not None] for c in CRITERES}
        totaux = [sum(_note(n.get(c)) for c in CRITERES) for n in notees
                  if all(_note(n.get(c)) is not None for c in CRITERES)]
        avis = [n.get("utilisable", "").strip().lower() for n in notees if n.get("utilisable", "").strip()]

        def moy(liste, cle):
            vals = [m[cle] for m in liste if m.get(cle) is not None]
            return round(mean(vals), 1) if vals else None

        lignes.append({
            "config": cfg, "cas_ok": f"{len(ok)}/{len(ms)}",
            "note_enseignant_sur_10": round(mean(totaux), 1) if totaux else None,
            **{f"moy_{c}": round(mean(v), 2) if v else None for c, v in par_critere.items()},
            "acceptation_pct": round(100 * sum(a in ("oui", "modif") for a in avis) / len(avis), 1) if avis else None,
            "rejet_pct": round(100 * sum(a == "non" for a in avis) / len(avis), 1) if avis else None,
            "productions_notees": len(totaux),
            "pb_taux_verif_ok_pct": moy(pb, "taux_verif_ok_pct"),
            "pb_erreurs_restantes": moy(pb, "erreurs_restantes"),
            "pb_duree_s": moy(pb, "duree_generation_s"),
            "qc_notion_trouvee_pct": moy(qc, "notion_trouvee_dans_le_cours_pct"),
            "qc_doublons": moy(qc, "quasi_doublons"),
            "qc_duree_par_question_s": moy(qc, "duree_par_question_s"),
            "duree_extraction_s": moy(ok, "duree_extraction_s"),
        })

    colonnes = list(lignes[0]) if lignes else []
    with open(run / "synthese.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=colonnes, delimiter=";")
        w.writeheader()
        w.writerows(lignes)
    md = ["# Synthèse du banc\n", f"Lancement : `{run.name}`\n",
          "| " + " | ".join(colonnes) + " |", "|" + "---|" * len(colonnes)]
    md += ["| " + " | ".join("—" if l[c] is None else str(l[c]) for c in colonnes) + " |" for l in lignes]
    erreurs = [m for m in mesures if m["statut"] != "ok"]
    if erreurs:
        md.append("\n## Cas non aboutis\n")
        md += [f"- {m['config']} / {m['cas']} : {m['statut']} {m.get('erreur', '')}" for m in erreurs]
    md.append("\nLégende : `pb_` = problèmes, `qc_` = questions de cours, durées en secondes, "
              "notes enseignant sur 2 par critère (total sur 10).")
    _ecrire(run / "synthese.md", "\n".join(md))
    return run / "synthese.md"
