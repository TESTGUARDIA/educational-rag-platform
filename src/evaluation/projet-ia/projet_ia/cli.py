"""
cli.py
======
Ligne de commande unique :  python -m projet_ia <commande> [options]

Commandes :
    verifier              Vérifier une réponse (interactif, démo, ou en une ligne)
    calibrer              Entraîner les poids de la vérification
    detection preparer    Calculer une fois pour toutes les features du dataset (cache)
    detection genetique   Chercher le meilleur MLP de chaque analyse + entraînement final
    detection entrainer   Entraînement final (fusion comprise) + test + rapport
    detection predire     Prédire si un paragraphe est écrit par une IA ou un humain

Les commandes de détection acceptent --langue en|fr|multi (anglais, français, multilingue).

Aide détaillée :  python -m projet_ia <commande> --help
"""

import argparse
import warnings

from . import config


def _construire_parseur() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m projet_ia",
        description="Vérification de réponses et détection de textes écrits par une IA.",
    )
    sub = parser.add_subparsers(dest="commande", required=True)

    # ── verifier ──────────────────────────────────────────────────────────────
    p = sub.add_parser("verifier", help="Vérifier une réponse d'élève",
                       description="Sans option : mode interactif. "
                                   "Avec --question/--reponse/--correction : une seule évaluation.")
    p.add_argument("--question")
    p.add_argument("--reponse")
    p.add_argument("--correction")
    p.add_argument("--demo", action="store_true", help="Évaluer les exemples de démonstration")

    # ── calibrer ──────────────────────────────────────────────────────────────
    p = sub.add_parser("calibrer", help="Calibrer les poids de la vérification",
                       description="Régression logistique par descente de gradient. "
                                   "Génère models/poids_calibres.json.")
    p.add_argument("--dataset", default=str(config.CHEMIN_META_DATASET),
                   help="CSV labellisé (défaut : %(default)s)")
    p.add_argument("--sortie", default=str(config.CHEMIN_POIDS_CALIBRES),
                   help="Fichier de poids généré (défaut : %(default)s)")
    p.add_argument("--lr", type=float, default=0.1, help="Taux d'apprentissage (défaut : 0.1)")
    p.add_argument("--epochs", type=int, default=5000, help="Nombre d'epochs (défaut : 5000)")
    p.add_argument("--l2", type=float, default=0.01, help="Régularisation L2 (défaut : 0.01)")
    p.add_argument("--k_fold", type=int, default=5, help="Folds de validation croisée (défaut : 5)")
    p.add_argument("--seuil", type=float, default=0.5, help="Seuil de décision (défaut : 0.5)")

    # ── detection ─────────────────────────────────────────────────────────────
    det = sub.add_parser("detection", help="Détection de textes écrits par une IA")
    ds = det.add_subparsers(dest="action", required=True)

    langues = list(config.LANGUES)

    def ajouter_langue(p, defaut=config.LANGUE_PAR_DEFAUT, extra=""):
        p.add_argument("--langue", choices=langues + (["auto"] if extra == "auto" else []),
                       default=defaut, help="Langue des textes (défaut : %(default)s)")

    p = ds.add_parser("preparer", help="Pré-calculer les features (embeddings, style) du dataset",
                      description="Étape lente, faite UNE fois par langue, reprenable si interrompue.")
    p.add_argument("--dataset", default=str(config.CHEMIN_DATASET_DETECTION),
                   help=".parquet / .jsonl / .csv (défaut : %(default)s)")
    ajouter_langue(p)
    p.add_argument("--cache", help="Dossier du cache (défaut : data/cache/<langue>)")
    p.add_argument("--encodeur", help="Modèle Hugging Face (défaut : celui de la langue)")
    p.add_argument("--phrases-train", type=int, default=config.PHRASES_PAR_TEXTE_TRAIN,
                   help="Phrases gardées par paragraphe pour entraîner le modèle-phrase")
    p.add_argument("--max-train", type=int, help="Limiter le split train (essais rapides)")
    p.add_argument("--max-val", type=int)
    p.add_argument("--max-test", type=int)
    p.add_argument("--n-jobs", type=int, default=1, help="Processus pour les features de style")
    p.add_argument("--chunk", type=int, default=10_000, help="Paragraphes par bloc sauvegardé")

    p = ds.add_parser("genetique", help="Algorithme génétique (un MLP par analyse) + entraînement final")
    ajouter_langue(p)
    p.add_argument("--cache", help="Dossier du cache (défaut : data/cache/<langue>)")
    p.add_argument("--composants", nargs="+", default=["paragraphe", "phrase", "interphrases"],
                   choices=["paragraphe", "phrase", "interphrases"],
                   help="Analyses à optimiser (défaut : les trois)")
    p.add_argument("--population", type=int, default=30)
    p.add_argument("--elite", type=int, default=10)
    p.add_argument("--lot-train", type=int, default=50_000,
                   help="Exemples de train par individu et par génération")
    p.add_argument("--taille-val", type=int, default=20_000)
    p.add_argument("--epochs-individu", type=int, default=5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--poids-rapidite", type=float, default=0.005,
                   help="Accuracy perdue par facteur 10 de temps d'entraînement : la vitesse "
                        "ne départage que des modèles de précision proche (défaut : %(default)s)")
    p.add_argument("--sans-entrainement-final", action="store_true",
                   help="S'arrêter après la recherche des meilleurs génomes")

    def ajouter_fpr(p):
        p.add_argument("--fpr-cible", type=float, default=config.FPR_CIBLE_DETECTION,
                       help="Part maximale de textes humains classés IA, qui fixe le seuil "
                            "(défaut : %(default)s)")

    ajouter_fpr(p)

    p = ds.add_parser("entrainer", help="Entraînement final + fusion + évaluation test + rapport",
                      description="Utilise les génomes de models/<langue>/ (sinon des génomes par défaut).")
    ajouter_langue(p)
    p.add_argument("--cache", help="Dossier du cache (défaut : data/cache/<langue>)")
    p.add_argument("--epochs", type=int, default=30)
    ajouter_fpr(p)

    p = ds.add_parser("predire", help="Prédire IA / humain pour un paragraphe")
    p.add_argument("texte", help="Un paragraphe (de préférence 3 phrases ou plus)")
    ajouter_langue(p, defaut="auto", extra="auto")
    p.add_argument("--modele", help="Forcer un fichier .pkl (désactive le choix par langue)")
    p.add_argument("--seuil", type=float,
                   help="Forcer le seuil de décision (défaut : celui appris à l'entraînement)")

    return parser


def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    args = _construire_parseur().parse_args(argv)

    # Les imports sont faits ici (et non en haut du fichier) pour que
    # `--help` reste instantané et n'exige pas torch / spaCy.
    if args.commande == "verifier":
        from .verification.interactif import (
            evaluer_et_afficher, lancer_demo, mode_interactif)
        if args.demo:
            lancer_demo()
        elif args.question and args.reponse and args.correction:
            evaluer_et_afficher(args.question, args.reponse, args.correction)
        else:
            mode_interactif()

    elif args.commande == "calibrer":
        from .calibration import calibrer
        calibrer(chemin_dataset=args.dataset, chemin_sortie=args.sortie,
                 taux_apprentissage=args.lr, epochs=args.epochs, l2=args.l2,
                 k_fold=args.k_fold, seuil_decision=args.seuil)

    elif args.commande == "detection":
        _commande_detection(args)


def _commande_detection(args) -> None:
    from . import config as cfg

    if args.action != "predire":
        cache = args.cache or str(cfg.CheminsDetection(args.langue).cache)

    if args.action == "preparer":
        from .detection.cache import construire_cache, construire_cache_flux
        from .detection.donnees import charger_dataset
        if args.dataset.lower().endswith(".parquet") and not (args.max_train or args.max_val
                                                               or args.max_test):
            # Dataset complet en Parquet : lecture en flux, mémoire indépendante de sa taille
            construire_cache_flux(args.dataset, cache, langue=args.langue,
                                  nom_encodeur=args.encodeur, taille_chunk=args.chunk,
                                  n_jobs=args.n_jobs, phrases_par_texte_train=args.phrases_train)
            return
        df = charger_dataset(args.dataset, langue=args.langue if args.langue != "multi" else None)
        print(f"{len(df):,} textes : "
              f"{(df['split'] == 'train').sum():,} train / {(df['split'] == 'val').sum():,} val / "
              f"{(df['split'] == 'test').sum():,} test")
        construire_cache(df, cache, langue=args.langue, nom_encodeur=args.encodeur,
                         taille_chunk=args.chunk, n_jobs=args.n_jobs,
                         phrases_par_texte_train=args.phrases_train,
                         max_par_split={"train": args.max_train, "val": args.max_val,
                                        "test": args.max_test})

    elif args.action == "genetique":
        from .detection.entrainement import entrainer_final
        from .detection.evolution import algorithme_genetique
        genomes = {c: algorithme_genetique(
                       cache, composant=c, taille_population=args.population,
                       taille_elite=args.elite, taille_lot_train=args.lot_train,
                       taille_val=args.taille_val, epochs_individu=args.epochs_individu,
                       patience=args.patience, max_generations=args.generations,
                       poids_rapidite=args.poids_rapidite)
                   for c in args.composants}
        if not args.sans_entrainement_final:
            entrainer_final(genomes, cache, fpr_cible=args.fpr_cible)

    elif args.action == "entrainer":
        from .detection.entrainement import entrainer_final
        entrainer_final(None, cache, epochs_max=args.epochs, fpr_cible=args.fpr_cible)

    elif args.action == "predire":
        from .detection.prediction import predire_depuis_fichier
        r = predire_depuis_fichier(args.texte, args.langue, args.modele, seuil=args.seuil)
        print(f"Prédiction : {r['prediction']}  (probabilité IA {r['probabilite_ia']:.2%}, "
              f"seuil {r['seuil']:.2%}, langue : {r['langue']})")
        print("  Analyses : " + " | ".join(f"{k} {v:.0%}" for k, v in r["composants"].items()))
        for ph in r["phrases"]:
            print(f"   {ph['probabilite_ia']:>5.0%}  {ph['texte'][:90]}")
        if "avertissement" in r:
            print(f"Attention : {r['avertissement']}")

if __name__ == "__main__":
    main()
