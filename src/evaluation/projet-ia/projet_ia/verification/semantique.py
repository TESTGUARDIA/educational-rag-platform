"""
verification/semantique.py
==========================
Pipeline sémantique : NLI + embeddings CamemBERT pondérés par catégorie
grammaticale + pénalités (négation, entités manquantes, concept hors champ).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..common.embeddings import embedding_cls as embedding
from ..common.models import get_camembert
from .modeles import nlp, nli
from .question import resoudre_coreferences
from .utilitaires import POIDS_POS, detecter_negation, mots_importants


def embedding_pondere(texte: str, poids_overrides: dict | None = None) -> np.ndarray:
    """
    Calcule un embedding pondéré d'un texte.
    Chaque token CamemBERT reçoit un poids selon sa catégorie grammaticale (POIDS_POS).
    Les mots grammaticaux (déterminants, pronoms…) sont ignorés.
    poids_overrides permet de sur-pondérer certaines catégories selon la question.
    """
    import torch
    tokenizer, modele = get_camembert()

    doc    = nlp(texte)
    inputs = tokenizer(
        texte, return_tensors="pt", truncation=True,
        max_length=512, return_offsets_mapping=False
    )
    with torch.no_grad():
        outputs = modele(**inputs)

    hidden     = outputs.last_hidden_state.squeeze(0).numpy()  # (n_tokens, 768)
    tokens_cam = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
    vecteurs   = []
    poids_lst  = []

    idx_cam = 1  # on ignore le token [CLS]
    for token_spacy in doc:
        if idx_cam >= len(tokens_cam) - 1:
            break

        poids     = POIDS_POS.get(token_spacy.pos_, 0.0)
        subtokens = []

        # Alignement spaCy ↔ CamemBERT (SentencePiece, ▁ = début de mot)
        mot_reconstruit = ""
        while idx_cam < len(tokens_cam) - 1:
            st = tokens_cam[idx_cam].replace("▁", "")
            mot_reconstruit += st
            subtokens.append(idx_cam)
            idx_cam += 1
            if mot_reconstruit.lower() == token_spacy.text.lower():
                break
            if len(mot_reconstruit) > len(token_spacy.text):
                break

        if subtokens and poids > 0:
            if poids_overrides and token_spacy.pos_ in poids_overrides:
                poids = poids_overrides[token_spacy.pos_]
            vecteurs.append(hidden[subtokens].mean(axis=0))
            poids_lst.append(poids)

    if not vecteurs:
        return hidden[0]  # fallback : token [CLS]

    return np.average(np.array(vecteurs), axis=0, weights=np.array(poids_lst))


def sens_global_mots_importants(texte: str,
                                 pos_prioritaires: list[str] | None = None) -> np.ndarray:
    """
    Calcule l'embedding du "sens global" d'un texte en ne gardant que
    les lemmes des mots importants, indépendamment de la formulation.
    Ex: "L'astre du jour brille" → embedding de "astre jour briller"
    """
    doc    = nlp(texte)
    lemmes = [
        t.lemma_ for t in doc
        if (pos_prioritaires is None or t.pos_ in pos_prioritaires)
        and POIDS_POS.get(t.pos_, 0.0) > 0
        and not t.is_stop
    ]
    if not lemmes:
        return embedding(texte)  # fallback
    return embedding(" ".join(lemmes))


def similarite_sens_global(texte1: str, texte2: str,
                             pos_prioritaires: list[str] | None = None) -> float:
    """Compare le sens global de deux textes (ne regarde que les mots importants)."""
    emb1 = sens_global_mots_importants(texte1, pos_prioritaires).reshape(1, -1)
    emb2 = sens_global_mots_importants(texte2, pos_prioritaires).reshape(1, -1)
    return float(cosine_similarity(emb1, emb2)[0][0])


def similarite_concept(mot: str, concept: str) -> float:
    """Calcule la similarité sémantique CamemBERT entre un mot et un concept."""
    emb_mot     = embedding(mot).reshape(1, -1)
    emb_concept = embedding(concept).reshape(1, -1)
    return float(cosine_similarity(emb_mot, emb_concept)[0][0])


def comparer_semantique(reponse: str, correction: str,
                         analyse_q: dict | None = None,
                         question: str = "") -> dict:
    """
    Compare deux réponses textuelles via un pipeline en 4 étapes :
      0. Résolution des coréférences ("C'est Paris" → "<sujet> est Paris")
      1. NLI : cohérence logique globale réponse ↔ correction
      2. Similarité sémantique pondérée + sens global (mots importants)
      3. Pénalités ciblées (négation, entités manquantes, concept hors champ)
         — neutralisées si le sens global est suffisant

    Args:
        reponse    : texte de la réponse à évaluer
        correction : texte de la correction attendue
        analyse_q  : résultat de analyser_question() — améliore la précision
        question   : question originale (pour la résolution des coréférences)

    Returns:
        dict avec mode, score (0-1), correct (bool), et détail complet
    """
    # ── Étape 0 : résolution des coréférences ─────────────────────────────────
    caractere = analyse_q.get("caractere_question", "ouverte") if analyse_q else "ouverte"
    reponse_resolue = (
        resoudre_coreferences(reponse, question)
        if question and caractere != "fermée"
        else reponse
    )

    # ── Construction des poids overrides selon la question ────────────────────
    poids_overrides = None
    echec_immediat  = False

    if analyse_q:
        pos_prio = analyse_q["pos_prioritaires"]
        multi    = analyse_q["multiplicateur"]
        poids_overrides = {pos: POIDS_POS.get(pos, 1.0) * multi for pos in pos_prio}

        doc_cor = nlp(correction)
        doc_rep = nlp(reponse_resolue)
        entites_cles = [t.text.lower() for t in doc_cor if t.pos_ in pos_prio and not t.is_stop]
        entites_rep  = [t.text.lower() for t in doc_rep]

        if entites_cles:
            taux = sum(1 for e in entites_cles if e in entites_rep) / len(entites_cles)
            if taux == 0.0 and multi >= 4.0:
                echec_immediat = True

    # ── Échec immédiat : entité clé absente ───────────────────────────────────
    if echec_immediat:
        return {
            "mode": "sémantique", "score": 0.0, "correct": False,
            "detail": {
                "echec_immediat":       True,
                "raison":               "Entité clé de la correction absente de la réponse",
                "reponse_resolue":      reponse_resolue,
                "mots_cles_correction": mots_importants(correction),
                "mots_cles_reponse":    mots_importants(reponse_resolue),
            }
        }

    # ── Étape 1 : NLI ─────────────────────────────────────────────────────────
    res_nli = nli(
        reponse_resolue,
        candidate_labels=["entailment", "neutral", "contradiction"],
        hypothesis_template="Cette affirmation est correcte : {}"
    )
    scores_nli = dict(zip(res_nli["labels"], res_nli["scores"]))
    s_nli = (scores_nli.get("entailment", 0) * 1.0
             + scores_nli.get("neutral", 0) * 0.5
             + scores_nli.get("contradiction", 0) * 0.0)

    # ── Étape 2 : Similarité pondérée ─────────────────────────────────────────
    emb_rep = embedding_pondere(reponse_resolue, poids_overrides).reshape(1, -1)
    emb_cor = embedding_pondere(correction, poids_overrides).reshape(1, -1)
    s_sim   = float(cosine_similarity(emb_rep, emb_cor)[0][0])

    # Sens global (NLI reponse vs correction directement)
    pos_prio_sens = analyse_q["pos_prioritaires"] if analyse_q else None
    s_sim_mots    = similarite_sens_global(reponse_resolue, correction, pos_prio_sens)

    res_nli_global = nli(
        reponse_resolue,
        candidate_labels=["cohérent avec", "contradictoire avec", "sans rapport avec"],
        hypothesis_template="Cette réponse est {} la correction attendue : " + correction[:200]
    )
    scores_nli_g = dict(zip(res_nli_global["labels"], res_nli_global["scores"]))
    s_nli_global = (scores_nli_g.get("cohérent avec", 0) * 1.0
                    + scores_nli_g.get("sans rapport avec", 0) * 0.3
                    + scores_nli_g.get("contradictoire avec", 0) * 0.0)

    s_sens_global = s_nli_global * 0.70 + s_sim_mots * 0.30

    # ── Étape 3 : Pénalités ───────────────────────────────────────────────────

    # Pénalité négation
    neg_reponse    = detecter_negation(reponse_resolue)
    neg_correction = detecter_negation(correction)
    penalite_neg   = 0.1 if (neg_reponse and not neg_correction) else 0.0

    # Pénalité entités manquantes
    penalite_entites    = 0.0
    taux_correspondance = 1.0
    if analyse_q and poids_overrides:
        pos_prio = analyse_q["pos_prioritaires"]
        multi    = analyse_q["multiplicateur"]
        doc_cor  = nlp(correction)
        doc_rep  = nlp(reponse)
        entites_cles = [t.text.lower() for t in doc_cor if t.pos_ in pos_prio and not t.is_stop]
        entites_rep  = [t.text.lower() for t in doc_rep]
        if entites_cles and multi >= 4.0:
            nb_trouvees         = sum(1 for e in entites_cles if e in entites_rep)
            taux_correspondance = nb_trouvees / len(entites_cles)
            penalite_entites    = (1.0 - taux_correspondance) * 0.8

    # Pénalité concept hors champ
    penalite_concept = 0.0
    info_concept     = {}

    if analyse_q and analyse_q.get("concept_cle"):
        concept  = analyse_q["concept_cle"]
        pos_prio = analyse_q["pos_prioritaires"]
        doc_rep  = nlp(reponse_resolue)
        doc_cor  = nlp(correction)

        mots_cor = [t.text for t in doc_cor if t.pos_ in pos_prio and not t.is_stop]
        mots_rep = [t.text for t in doc_rep if t.pos_ in pos_prio and not t.is_stop]

        if mots_cor and mots_rep:
            SEUIL = 0.75
            couvertures = []
            for mot_cor in mots_cor:
                tok_cor  = nlp(mot_cor)[0]
                sims     = []
                for mot_rep in mots_rep:
                    tok_rep = nlp(mot_rep)[0]
                    if tok_cor.has_vector and tok_rep.has_vector:
                        sims.append(tok_cor.similarity(tok_rep))
                    else:
                        sims.append(similarite_concept(mot_rep, mot_cor))
                couvertures.append(max(sims) if sims else 0.0)

            score_moy_couv = sum(couvertures) / len(couvertures)

            sim_concept_max = max(
                (nlp(m)[0].similarity(nlp(concept["lemme"])[0])
                 if nlp(m)[0].has_vector and nlp(concept["lemme"])[0].has_vector
                 else similarite_concept(m, concept["lemme"]))
                for m in mots_rep
            )

            if sim_concept_max <= 0.5:
                penalite_concept = 0.0
            elif sim_concept_max >= 0.85:
                penalite_concept = 0.0
            else:
                penalite_concept = (1.0 - score_moy_couv) * 0.4

            info_concept = {
                "concept_attendu":  concept["mot"],
                "sim_concept":      round(sim_concept_max, 4),
                "penalite_concept": round(penalite_concept, 4),
            }

        elif mots_cor and not mots_rep:
            penalite_concept = 0.8
            info_concept = {"concept_attendu": concept["mot"], "penalite_concept": 0.8}

    # ── Combinaison selon le caractère de la question ─────────────────────────
    if caractere == "fermée":
        score_sem_combine      = s_sim * 0.85 + s_sens_global * 0.15
        facteur_neutralisation = 0.0
        poids_nli, poids_sem   = 0.20, 0.60
        penalite_concept       = 0.0
    elif caractere == "semi-ouverte":
        score_sem_combine      = s_sens_global * 0.55 + s_sim * 0.45
        facteur_neutralisation = max(0.0, (s_sens_global - 0.50) / 0.50) if s_sens_global >= 0.50 else 0.0
        poids_nli, poids_sem   = 0.20, 0.70
    else:
        score_sem_combine      = s_sens_global * 0.55 + s_sim * 0.45
        facteur_neutralisation = max(0.0, (s_sens_global - 0.50) / 0.50) if s_sens_global >= 0.50 else 0.0
        poids_nli, poids_sem   = 0.20, 0.70

    penalite_concept = penalite_concept * (1.0 - facteur_neutralisation)
    if caractere == "fermée":
        penalite_entites = penalite_entites * (1.0 - facteur_neutralisation)
    else:
        penalite_entites = penalite_entites * (1.0 - facteur_neutralisation) * 0.3

    score = max(0.0, min(1.0,
        s_nli                * poids_nli
        + score_sem_combine  * poids_sem
        - penalite_neg       * 0.5
        - penalite_entites
        - penalite_concept
    ))

    return {
        "mode":    "sémantique",
        "score":   round(score, 4),
        "correct": score >= 0.6,
        "detail": {
            "nli":                      round(s_nli, 4),
            "similarite_ponderee":      round(s_sim, 4),
            "sens_global_mots_cles":    round(s_sens_global, 4),
            "score_semantique_combine": round(score_sem_combine, 4),
            "negation":                 neg_reponse,
            "penalite_neg":             penalite_neg,
            "penalite_entites":         round(penalite_entites, 4),
            "taux_entites_trouvees":    round(taux_correspondance, 4),
            "penalite_concept":         round(penalite_concept, 4),
            "info_concept":             info_concept,
            "type_question":            analyse_q["type_question"] if analyse_q else "inconnu",
            "caractere_question":       caractere,
            "pos_prioritaires":         analyse_q["pos_prioritaires"] if analyse_q else [],
            "reponse_resolue":          reponse_resolue,
            "mots_cles_reponse":        mots_importants(reponse_resolue),
            "mots_cles_correction":     mots_importants(correction),
        }
    }
