"""
detection/entrainement.py
=========================
Entraînement FINAL de tous les composants, apprentissage de la fusion, puis
évaluation UNIQUE sur le split `test` et génération du rapport.

Déroulé :
  1. chaque composant (paragraphe, phrase, interphrases) est entraîné sur `train`
     (arrêt anticipé sur `val`) ;
  2. on calcule leurs log-odds sur `val` et on y apprend les poids de fusion par
     descente de gradient, puis le SEUIL de décision : le plus bas qui garde au plus
     `fpr_cible` des textes humains de `val` classés « IA » (faux positifs) ;
  3. le split `test`, jamais utilisé jusque-là, sert une seule fois à mesurer chaque
     composant ET la fusion (au seuil choisi, et à 0.5 pour comparaison).
Les poids de fusion sont appris sur des prédictions de `val` que les composants
n'ont pas vues à l'entraînement (sinon la fusion ferait confiance au composant
qui a mémorisé son train).
"""

import json
import pickle
from datetime import datetime


from .. import config
from ..calibration.regression import sigmoid
from ..common.fichiers import creer_dossier_parent
from .cache import lire_info_cache
from .composants import COMPOSANTS, charger_donnees, genome_par_defaut
from .fusion import (agreger_phrases, assembler_entrees, entrainer_fusion, logit,
                     predire_fusion)
from .mlp import (GenomeMLP, assembler_features, calculer_metriques, entrainer_mlp,
                  metriques_par_groupe, predire_proba, seuil_pour_fpr)


def _features(d: dict, genome: GenomeMLP):
    return assembler_features(d["emb"], d["style"], d["coh"], genome, d["ppl"])


def charger_genomes(langue: str, genomes: dict | None = None) -> dict:
    """Génome de chaque composant : celui fourni, sinon models/<langue>/genome_<composant>.json,
    sinon le génome par défaut."""
    genomes = dict(genomes or {})
    chemins = config.CheminsDetection(langue)
    for c in COMPOSANTS:
        if c in genomes:
            continue
        fichier = chemins.genome(c)
        if fichier.exists():
            with open(fichier, encoding="utf-8") as f:
                genomes[c] = GenomeMLP.from_dict(json.load(f))
        else:
            print(f"[{c}] pas de génome trouvé ({fichier.name}) : génome par défaut utilisé.")
            genomes[c] = genome_par_defaut(c)
    return genomes


def _sorties_split(modeles: dict, genomes: dict, dossier, split: str) -> dict:
    """Log-odds de tous les composants sur un split, avec labels et métadonnées."""
    para = charger_donnees(dossier, "paragraphe", split)
    inter = charger_donnees(dossier, "interphrases", split)
    phr = charger_donnees(dossier, "phrase", split)
    z_phrases = logit(predire_proba(modeles["phrase"], _features(phr, genomes["phrase"])))
    z = {
        "paragraphe": logit(predire_proba(modeles["paragraphe"],
                                          _features(para, genomes["paragraphe"]))),
        "interphrases": logit(predire_proba(modeles["interphrases"],
                                            _features(inter, genomes["interphrases"]))),
        **agreger_phrases(z_phrases, phr["idx_texte"], len(para["y"])),
    }
    return {"z": z, "y": para["y"], "meta": para["meta"],
            "z_phrases": z_phrases, "y_phrases": phr["y"]}


def entrainer_final(genomes: dict | None = None, dossier_cache=None,
                    chemin_modele=None, chemin_rapport=None,
                    epochs_max: int = 30, patience: int = 3, seed: int = 42,
                    max_phrases_train: int | None = 1_000_000,
                    fpr_cible: float = config.FPR_CIBLE_DETECTION,
                    entraineur=entrainer_mlp) -> dict:
    """
    Entraîne les trois composants, apprend la fusion et le seuil, évalue sur `test`,
    sauvegarde le modèle (.pkl) et le rapport (.json).

    Args:
        genomes           : dict composant -> GenomeMLP (sinon fichiers de models/<langue>/
                            produits par l'algorithme génétique, sinon génomes par défaut)
        dossier_cache     : cache de la langue visée (obligatoire en pratique)
        max_phrases_train : nombre maximum de phrases pour entraîner le modèle-phrase (mémoire)
        fpr_cible         : part maximale de textes humains de `val` classés « IA »,
                            qui fixe le seuil de décision (ex. 0.01 = 1 %)
    Returns:
        le rapport (dict)
    """
    info = lire_info_cache(dossier_cache)
    chemins = config.CheminsDetection(info["langue"])
    chemin_modele = chemin_modele or chemins.modele
    chemin_rapport = chemin_rapport or chemins.rapport
    genomes = charger_genomes(info["langue"], genomes)

    # ── 1. Entraînement de chaque composant ──
    modeles, details = {}, {}
    for c in COMPOSANTS:
        print(f"\n[{c}] Entraînement sur le split train...")
        train = charger_donnees(dossier_cache, c, "train",
                                max_phrases_train if c == "phrase" else None, seed)
        val = charger_donnees(dossier_cache, c, "val", None, seed)
        res = entraineur(genomes[c], _features(train, genomes[c]), train["y"],
                         _features(val, genomes[c]), val["y"],
                         epochs_max=epochs_max, patience=patience, seed=seed)
        modeles[c] = res
        details[c] = {"genome": genomes[c].to_dict(), "architecture": list(genomes[c].couches()),
                      "n_exemples_train": int(len(train["y"])),
                      "epoch_optimal": res["epoch_optimal"],
                      "temps_secondes": round(res["temps"], 1), "historique": res["historique"]}
        print(f"[{c}] accuracy val {res['accuracy_val']:.4f} (époque {res['epoch_optimal']})")
        del train, val

    # ── 2. Fusion : poids appris sur la validation ──
    sortie_val = _sorties_split(modeles, genomes, dossier_cache, "val")
    print("\nApprentissage des poids de fusion (descente de gradient)...")
    fusion = entrainer_fusion(assembler_entrees(sortie_val["z"]), sortie_val["y"])
    p_val = predire_fusion(fusion, assembler_entrees(sortie_val["z"]))
    seuil = seuil_pour_fpr(sortie_val["y"], p_val, fpr_cible)
    print(f"Seuil de décision : {seuil:.4f} (au plus {fpr_cible:.1%} de faux positifs sur val)")

    # ── 3. Évaluation sur le jeu de test : une seule fois, à la toute fin ──
    sortie_test = _sorties_split(modeles, genomes, dossier_cache, "test")
    y_test, meta_test = sortie_test["y"], sortie_test["meta"]
    p_fusion = predire_fusion(fusion, assembler_entrees(sortie_test["z"]))

    for c, z in (("paragraphe", "paragraphe"), ("interphrases", "interphrases"),
                 ("phrase", "phrases_moy")):
        details[c]["test"] = calculer_metriques(y_test, sigmoid(sortie_test["z"][z]))

    rapport = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "langue": info["langue"], "cache": info,
        "tailles": {s: int(len(sortie["y"])) for s, sortie in
                    (("val", sortie_val), ("test", sortie_test))},
        "composants": details,
        "fusion": {"poids_ia": fusion["poids_ia"], "poids_humain": fusion["poids_humain"],
                   "biais": fusion["biais"], "historique_cout": fusion["historique_cout"],
                   "validation": calculer_metriques(sortie_val["y"], p_val, seuil)},
        "seuil": {"valeur": seuil, "fpr_cible": fpr_cible,
                  "methode": "plus bas seuil gardant au plus fpr_cible de faux positifs sur val"},
        "test": calculer_metriques(y_test, p_fusion, seuil),
        "test_seuil_0_5": calculer_metriques(y_test, p_fusion, 0.5),
        "test_niveau_phrase": calculer_metriques(sortie_test["y_phrases"],
                                                 sigmoid(sortie_test["z_phrases"])),
        "test_par_domaine": metriques_par_groupe(meta_test, y_test, p_fusion, "domaine", seuil),
        "test_par_longueur": metriques_par_groupe(meta_test, y_test, p_fusion, "longueur", seuil),
        "test_detection_par_generateur": metriques_par_groupe(
            meta_test, y_test, p_fusion, "generateur", seuil),
        "remarque": ("Les poids de fusion, le seuil et l'arrêt anticipé utilisent le split val ; "
                     "le split test n'est lu qu'une fois, ici."),
    }

    creer_dossier_parent(chemin_modele)
    with open(chemin_modele, "wb") as f:
        pickle.dump({"version": 2, "composants": {
                        c: {"modele": modeles[c]["modele"], "scaler": modeles[c]["scaler"],
                            "genome": genomes[c].to_dict()} for c in COMPOSANTS},
                     "fusion": fusion, "seuil": seuil, "fpr_cible": fpr_cible,
                     "cache": info}, f)
    creer_dossier_parent(chemin_rapport)
    with open(chemin_rapport, "w", encoding="utf-8") as f:
        json.dump(rapport, f, indent=2, ensure_ascii=False)

    afficher_rapport(rapport)
    print(f"\nModèle  : {chemin_modele}\nRapport : {chemin_rapport}")
    return rapport


def afficher_rapport(rapport: dict) -> None:
    """Résumé lisible du rapport d'entraînement."""
    t = rapport["test"]
    print("\n" + "=" * 66)
    print(f"  RAPPORT D'ENTRAÎNEMENT — langue : {rapport['langue']}")
    print("=" * 66)
    print("  Accuracy sur le jeu de TEST (paragraphes) :")
    for c, d in rapport["composants"].items():
        print(f"    {c:<14} seul : {d['test']['accuracy']:.4f}   (architecture {d['architecture']})")
    print(f"    {'FUSION':<14}      : {t['accuracy']:.4f} | précision {t['precision']:.4f} | "
          f"rappel {t['rappel']:.4f} | F1 {t['f1']:.4f} | AUC {t['auc']}")
    if "seuil" in rapport:
        s, t05 = rapport["seuil"], rapport["test_seuil_0_5"]
        print(f"  Seuil {s['valeur']:.4f} (cible : au plus {s['fpr_cible']:.1%} de faux positifs) : "
              f"faux positifs sur test {t['taux_faux_positifs']:.2%}, IA détectées {t['rappel']:.2%}")
        print(f"  (au seuil 0.5 : faux positifs {t05['taux_faux_positifs']:.2%}, "
              f"IA détectées {t05['rappel']:.2%})")
    print(f"  Niveau phrase (labels hérités du paragraphe) : accuracy "
          f"{rapport['test_niveau_phrase']['accuracy']:.4f}")

    fu = rapport["fusion"]
    print("\n  Poids de fusion appris (log-odds) :")
    print(f"    {'entrée':<14} {'poids IA':>10} {'poids humain':>14}")
    for n in fu["poids_ia"]:
        print(f"    {n:<14} {fu['poids_ia'][n]:>10.3f} {fu['poids_humain'][n]:>14.3f}")
    print(f"    biais : {fu['biais']:.3f}")

    for titre, cle in (("Par domaine", "test_par_domaine"), ("Par longueur", "test_par_longueur")):
        print(f"\n  {titre} (accuracy de la fusion) :")
        for nom, m in rapport[cle].items():
            print(f"    {nom:<14} {m['accuracy']:.4f}   (n={m['n']})")
    pires = list(rapport["test_detection_par_generateur"].items())[:5]
    if pires:
        print("\n  Générateurs IA les moins bien détectés :")
        for nom, m in pires:
            print(f"    {nom:<18} {m['accuracy']:.4f}   (n={m['n']})")
