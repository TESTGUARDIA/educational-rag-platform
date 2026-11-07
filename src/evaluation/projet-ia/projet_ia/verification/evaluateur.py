"""
verification/evaluateur.py
==========================
Point d'entrée principal : `evaluer_reponse(question, reponse, correction)`.

Disjonction de cas automatique :
  1. Question composite (plusieurs sous-questions) → évaluation multi-parties
  2. Correction purement mathématique              → pipeline numérique
  3. Correction hybride (texte + nombres)          → pipeline hybride
  4. Correction textuelle                          → pipeline sémantique
"""

from .numerique import (
    comparer_numerique,
    est_reponse_hybride,
    est_reponse_purement_mathematique,
)
from .poids import score_calibre
from .question import (
    analyser_question,
    decouper_reponse_correction,
    detecter_sous_questions,
)
from .semantique import comparer_semantique


def evaluer_reponse_multi(question: str, reponse: str, correction: str) -> dict | None:
    """
    Évalue une réponse multi-parties si la question contient plusieurs sous-questions.
    Score final = moyenne des scores partiels.
    Retourne None si la question est simple (pas de découpage nécessaire).
    """
    sous_questions = detecter_sous_questions(question)
    if len(sous_questions) <= 1:
        return None

    paires = decouper_reponse_correction(reponse, correction, len(sous_questions))
    resultats_partiels = []

    for sq, (rep_i, cor_i) in zip(sous_questions, paires):
        r_i = evaluer_reponse(sq + " ?", rep_i, cor_i)
        resultats_partiels.append({
            "sous_question":    sq + " ?",
            "reponse_partie":   rep_i,
            "correction_partie": cor_i,
            "score":            r_i["score"],
            "correct":          r_i["correct"],
            "mode":             r_i.get("mode", "?"),
        })

    scores = [r["score"] for r in resultats_partiels]
    return {
        "mode":               "multi-parties",
        "score":              round(sum(scores) / len(scores), 4),
        "correct":            all(r["correct"] for r in resultats_partiels),
        "n_parties":          len(sous_questions),
        "resultats_partiels": resultats_partiels,
        "detail": {
            "sous_questions":   sous_questions,
            "scores_partiels":  scores,
            "correct_partiels": [r["correct"] for r in resultats_partiels],
        }
    }


def evaluer_reponse(question: str, reponse: str, correction: str) -> dict:
    """
    Évalue si une réponse est correcte par rapport à une correction.

    Disjonction de cas automatique :
      1. Question composite (plusieurs sous-questions) → évaluation multi-parties
      2. Correction purement mathématique              → pipeline NUMÉRIQUE
      3. Correction hybride (texte + nombres)          → pipeline HYBRIDE
      4. Correction textuelle                          → pipeline SÉMANTIQUE

    Args:
        question   : la question posée (contexte)
        reponse    : la réponse à évaluer
        correction : la réponse attendue

    Returns:
        dict avec mode, score (0-1), correct (bool), type_question, et detail
    """
    # Cas 1 : question composite
    resultat_multi = evaluer_reponse_multi(question, reponse, correction)
    if resultat_multi is not None:
        resultat_multi["type_question"] = analyser_question(question)["type_question"]
        resultat_multi["verdict"] = _verdict_depuis_score(resultat_multi["score"])
        return resultat_multi

    analyse_q = analyser_question(question)

    # Cas 2 : réponse purement mathématique
    if est_reponse_purement_mathematique(reponse, correction):
        resultat = comparer_numerique(reponse, correction)

    # Cas 3 : réponse hybride (texte + chiffres)
    elif est_reponse_hybride(correction):
        resultat_sem = comparer_semantique(reponse, correction, analyse_q, question)
        resultat_num = comparer_numerique(reponse, correction)
        score_num    = resultat_num["score"]
        score_sem    = resultat_sem["score"]

        if score_num >= 0.95:
            # Nombres corrects → neutralise les pénalités sémantiques
            detail    = resultat_sem.get("detail", {})
            s_nli     = detail.get("nli", 0.5)
            s_sim     = detail.get("similarite_ponderee", 0.5)
            s_sens    = detail.get("sens_global_mots_cles", 0.5)
            score_pos = min(1.0, s_nli * 0.20 + (s_sens * 0.55 + s_sim * 0.45) * 0.70)
            score_hybride = round(max(score_sem, score_pos), 4)
        else:
            # Nombres incorrects → sémantique dominant + malus numérique
            score_hybride = round(score_sem * 0.80 + score_num * 0.20, 4)

        resultat = resultat_sem.copy()
        resultat["score"]   = score_hybride
        resultat["correct"] = score_hybride >= 0.6
        resultat["mode"]    = "hybride"
        resultat["detail"]["score_numerique_bonus"] = score_num

    # Cas 4 : réponse textuelle pure
    else:
        resultat = comparer_semantique(reponse, correction, analyse_q, question)
        # Affine avec le score calibré par descente de gradient (moyenne avec
        # le score manuel pour plus de robustesse — voir calibration_entrainement.py)
        detail = resultat.get("detail", {})
        sc_cal = score_calibre({
            "nli":                 detail.get("nli", 0.5),
            "similarite_ponderee": detail.get("similarite_ponderee", 0.5),
            "sens_global":         detail.get("sens_global_mots_cles", 0.5),
            "penalite_neg":        detail.get("penalite_neg", 0.0),
            "penalite_entites":    detail.get("penalite_entites", 0.0),
            "penalite_concept":    detail.get("penalite_concept", 0.0),
            "caractere_question":  detail.get("caractere_question", "ouverte"),
        })
        resultat["score"]   = round((resultat["score"] + sc_cal) / 2, 4)
        resultat["correct"] = resultat["score"] >= 0.6
        resultat["detail"]["score_calibre"] = round(sc_cal, 4)

    resultat["type_question"] = analyse_q["type_question"]
    resultat["verdict"] = _verdict_depuis_score(resultat["score"])
    return resultat


def _verdict_depuis_score(score: float) -> str:
    """Traduit un score [0,1] en verdict lisible."""
    if score >= 0.60:
        return "Correct"
    return "Incorrect"
