"""
Branchements du backend web vers les pôles Génération et Évaluation.

    generer_questions()  -> src/generation/rag.py     (pôle Génération)
    corriger_reponse()   -> projet_ia.verification    (pôle Évaluation)
    detecter_ia()        -> projet_ia.detection       (pôle Évaluation)

Les imports lourds (LangChain, PyTorch, transformers, spaCy) sont faits à la
première utilisation : le backend démarre, et ses tests tournent, sans eux.
Les tests remplacent ces trois fonctions par des doublures.
"""

import logging
import os
import sys
import threading
from pathlib import Path

log = logging.getLogger("testguardai.services")

# Racine du dépôt : src/web/back/services.py -> ../../..
RACINE = Path(__file__).resolve().parents[3]
DOSSIER_PROJET_IA = RACINE / "src" / "evaluation" / "projet-ia"

# Hors Docker, projet-ia n'est pas installé : on le rend importable depuis le dépôt.
if DOSSIER_PROJET_IA.is_dir() and str(DOSSIER_PROJET_IA) not in sys.path:
    sys.path.append(str(DOSSIER_PROJET_IA))

STATUT_ANALYSE = "analysé"
STATUT_NON_ANALYSE = "non_analysé"
STATUT_ERREUR = "erreur"

# Les modèles tiennent en mémoire une seule fois : on sérialise les appels pour
# éviter que deux requêtes simultanées ne doublent la mémoire utilisée (CPU).
_verrou_generation = threading.Lock()
_verrou_evaluation = threading.Lock()


# ─── GÉNÉRATION ────────────────────────────────────────────────────────────────
def generer_questions(pdf: bytes, nom_fichier: str, nb_questions: int) -> list[dict]:
    """Retourne [{"question": ..., "reponse": ...}] générés depuis le PDF (via Ollama)."""
    from src.generation.rag import ingest_and_generate_questions

    with _verrou_generation:
        brutes = ingest_and_generate_questions(pdf, nom_fichier, nb_questions=nb_questions)

    questions = []
    for q in brutes or []:
        if not isinstance(q, dict):
            continue
        enonce = str(q.get("question") or "").strip()
        reponse = str(q.get("reponse") or "").strip()
        if enonce:
            questions.append({"question": enonce, "reponse": reponse})
    return questions


# ─── CORRECTION ────────────────────────────────────────────────────────────────
def corriger_reponse(question: str, reponse: str, correction: str) -> dict:
    """Score [0, 1] d'une réponse d'élève par rapport à la réponse attendue."""
    if not (reponse or "").strip():
        return {"score": 0.0, "correct": False, "verdict": "Incorrect", "mode": "vide"}

    from projet_ia.verification import evaluer_reponse

    with _verrou_evaluation:
        r = evaluer_reponse(question=question, reponse=reponse, correction=correction)
    return {"score": float(r["score"]), "correct": bool(r["correct"]),
            "verdict": r.get("verdict"), "mode": r.get("mode")}


# ─── DÉTECTION IA ──────────────────────────────────────────────────────────────
def detecter_ia(textes: list[str]) -> dict:
    """
    Analyse les réponses d'une copie (une réponse = un paragraphe).

    Retourne toujours un dict avec `statut` :
      - "analysé"     : score (moyenne pondérée par le nombre de phrases), seuil
                        du modèle, phrases au-dessus du seuil ;
      - "non_analysé" : aucun modèle pour la langue de la copie, ou copie trop
                        courte pour identifier la langue (`raison` explique) ;
      - "erreur"      : échec technique (`raison` contient le message).
    """
    textes = [t.strip() for t in textes if t and t.strip()]
    if not textes:
        return {"statut": STATUT_NON_ANALYSE, "raison": "Copie vide."}

    try:
        from projet_ia.detection.langue import detecter_langue

        langue = detecter_langue("\n".join(textes))
        if langue is None:
            return {"statut": STATUT_NON_ANALYSE,
                    "raison": "Langue non identifiée (copie trop courte)."}

        from projet_ia.detection import predire_textes

        with _verrou_evaluation:
            resultats = predire_textes(textes, langue=langue)
    except FileNotFoundError:
        return {"statut": STATUT_NON_ANALYSE, "langue": langue,
                "raison": f"Aucun modèle de détection pour la langue « {langue} »."}
    except Exception as e:  # modèle illisible, téléchargement impossible…
        log.exception("Détection IA impossible")
        return {"statut": STATUT_ERREUR, "raison": f"{type(e).__name__} : {e}"}

    seuil = float(resultats[0]["seuil"])
    poids = [max(1, int(r.get("n_phrases") or 1)) for r in resultats]
    score = sum(r["probabilite_ia"] * p for r, p in zip(resultats, poids)) / sum(poids)
    phrases = [ph for r in resultats for ph in r.get("phrases", [])
               if ph["probabilite_ia"] >= seuil]
    avertissements = sorted({r["avertissement"] for r in resultats if r.get("avertissement")})
    return {"statut": STATUT_ANALYSE, "score": round(score, 4), "seuil": round(seuil, 4),
            "langue": langue, "phrases": phrases, "avertissements": avertissements}


# ─── PRÉCHARGEMENT ─────────────────────────────────────────────────────────────
def precharger_modeles() -> None:
    """
    Charge (et télécharge au premier lancement) tous les modèles d'évaluation,
    pour que la première copie analysée n'attende pas plusieurs minutes.
    Lancé dans un thread au démarrage si PRECHARGER_MODELES=1.
    """
    log.warning("Préchargement des modèles d'évaluation…")
    try:
        corriger_reponse("Quelle est la capitale de la France ?", "C'est Paris.",
                         "Paris est la capitale de la France.")
        detecter_ia(["This is a short text used to warm up the detection models. "
                     "It contains several sentences. Nothing more is needed here."])
        log.warning("Modèles d'évaluation prêts.")
    except Exception:
        log.exception("Préchargement des modèles impossible")


def demarrer_prechargement() -> None:
    if os.environ.get("PRECHARGER_MODELES", "0") == "1":
        threading.Thread(target=precharger_modeles, name="prechargement", daemon=True).start()
