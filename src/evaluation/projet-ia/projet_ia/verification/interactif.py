"""
verification/interactif.py
==========================
Mode interactif et démonstration de la vérification de réponses.
Lancé par : python -m projet_ia verifier
"""

from .evaluateur import evaluer_reponse

DEMOS = [
    {"question":   "Quelle est la capitale de la France ?",
     "reponse":    "La capitale de la France est Paris.",
     "correction": "Paris est la capitale de la France."},
    {"question":   "Qui a écrit Hamlet ?",
     "reponse":    "Hamlet a été écrit par Victor Hugo.",
     "correction": "Hamlet a été écrit par William Shakespeare."},
    {"question":   "Combien font 15 × 8 ?",
     "reponse":    "15 × 8 = 120",
     "correction": "Le résultat est 120."},
    {"question":   "Résous l'équation : ln(x) = 2",
     "reponse":    "x = e²",
     "correction": "ln(x) = 2 ⟹ x = e²"},
    {"question":   "Quel est le plus grand océan du monde et sa taille ?",
     "reponse":    "C'est l'océan Pacifique et il fait 165 000 000 km².",
     "correction": "C'est l'océan Pacifique et il fait 165 millions de km²."},
]


def afficher_resultat(r: dict) -> None:
    """Affiche le résultat d'evaluer_reponse() de façon lisible."""
    sc_bar = "█" * int(r["score"] * 30) + "░" * (30 - int(r["score"] * 30))
    print(f"\n  [{sc_bar}] {r['score']:.0%}")
    print(f"  Verdict    : {r['verdict']}")
    print(f"  Mode       : {r['mode']}")
    print(f"  Type Q     : {r.get('type_question', '?')}")
    d = r.get("detail", {})
    for k in ["nli", "similarite_ponderee", "sens_global_mots_cles",
              "penalite_neg", "penalite_entites", "penalite_concept",
              "score_calibre", "score_numerique_bonus"]:
        if k in d:
            print(f"  {k:<28}: {d[k]}")
    if r["mode"] == "multi-parties":
        for p in r.get("resultats_partiels", []):
            print(f"    Partie : {p['sous_question'][:50]}"
                  f"  → {p['score']:.0%}  ({'OK' if p['correct'] else 'KO'})")


def evaluer_et_afficher(question: str, reponse: str, correction: str) -> dict:
    """Évalue une réponse et affiche le détail. Retourne le résultat."""
    print(f"\n{'='*65}")
    print(f"  Q  : {question}")
    print(f"  R  : {reponse}")
    print(f"  C  : {correction}")
    r = evaluer_reponse(question, reponse, correction)
    afficher_resultat(r)
    return r


def lancer_demo() -> None:
    """Évalue les exemples de DEMOS."""
    for ex in DEMOS:
        evaluer_et_afficher(ex["question"], ex["reponse"], ex["correction"])


def mode_interactif() -> None:
    """Boucle interactive : saisie question / réponse / correction."""
    print("\n" + "=" * 65)
    print("  VÉRIFICATION DE RÉPONSE — Mode interactif")
    print("  Entrez 'q' pour quitter | 'demo' pour les exemples")
    print("=" * 65)

    while True:
        print()
        entree = input("Commande [demo/entrer/q] : ").strip().lower()
        if entree == "q":
            break
        if entree == "demo":
            lancer_demo()
            continue
        question   = input("Question   : ").strip()
        reponse    = input("Réponse    : ").strip()
        correction = input("Correction : ").strip()
        if not question or not reponse or not correction:
            print("  Champs manquants.")
            continue
        evaluer_et_afficher(question, reponse, correction)
