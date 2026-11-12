# Avancement par jalon

Source : *TestGuardAI — Plan d'action par jalon* (30 oct. 2026). Mettre ce fichier à
jour dans la PR qui fait changer une exigence de statut (voir le modèle de PR).

Statuts : ⬜ à faire · 🟨 en cours · ✅ fait · ⛔ bloqué

## Sécuriser et organiser (30 oct. – 6 nov.)

| Tâche | Statut | Remarque |
|---|---|---|
| Révoquer les clés Google vues dans `main` et la branche génération | 🟨 | À confirmer : 2 clés (`AIzaSyCV…`, `AIzaSyBO…`) restent dans l'historique public, ignorées par gitleaks via `.gitleaksignore` |
| Renommer les branches en `feature/web-vanilla`, `feature/generation`, `feature/evaluation` | ✅ | Sauvegardes : tags `backup-*`, `archive/*` |
| Créer `develop`, protéger `main`, taguer `archive/netlify-v0` | 🟨 | Protections de branche à activer (étape 7) |
| Nettoyer le dépôt | ✅ | |
| CODEOWNERS, modèle de PR, CI pytest + gitleaks | ✅ | gitleaks en CLI (l'action exige une licence pour une organisation) |
| Milestones J1 à J5, une issue par exigence | ⬜ | |
| `docs/` : CDC, état de l'art, avancement, traçabilité | 🟨 | CDC et état de l'art à déposer |
| Point tuteurs : J1 avec réserve, renégociations, Virna au §9, ENF-02 | ⬜ | |
| Demander les copies antérieures au 30/11/2022, réserver le GPU CRIStAL | ⬜ | |
| Trancher le versionnement des modèles (§8) | ✅ | Voir « Décisions » ci-dessous |

## J1 — Rattrapage (avant le 13 novembre) — ✅ atteint dans `develop`

Preuve attendue : un parcours complet upload, génération, réponse, puis note et score IA.

Génération via `src/generation/rag.py`, correction et détection via `src/evaluation/projet-ia`,
branchées dans `src/web/back/services.py` (WEB-01, COR-01, DET-05).

| Tâche | Statut | Remarque |
|---|---|---|
| Fusionner `projet-ia` et le code de génération dans `develop` | ✅ | `feature/j1-integration` fusionnée puis supprimée |
| `POST /devoirs/{id}/generer` et `POST /soumissions/{sid}/corriger-auto` | ✅ | |
| Brancher `predire_depuis_fichier` dans `/soumissions` et `/analyser-ia` (seuil du modèle, `non_analysé`) | ✅ | |
| Brancher `evaluer_reponse` et `ingest_and_generate_questions` (+ `nb_questions`) | ✅ | |
| Aligner `/stats` et les badges sur le seuil du modèle | ✅ | |
| `Dockerfile.api`, `requirements.txt`, `.env.example`, retirer `random` | ✅ | |
| `docker compose up` sur une machine vierge (ENF-10) | ⬜ | |

## J2 — Sécurité et données (30 novembre)

| Tâche | Statut |
|---|---|
| Collecte des copies et génération IA appariée en français (chemin critique) | ⬜ |
| WEB-02 : rôle administrateur | ⬜ |
| WEB-02 : tests pytest des trois critères d'acceptation | ⬜ |
| WEB-02 : httpOnly et PKCE, ou retirer l'affirmation | ⬜ |
| DET-02 : corpus français, filtres §8, splits par groupe, fiche descriptive | ⬜ |
| Jeu annoté de correction : ≥ 300 réponses sur 30 questions, double notation 20 % | ⬜ |
| ENF-02 : anonymisation (NER spaCy) avant stockage | ⬜ |
| ENF-03 : mots de passe de démonstration, gitleaks vert | ⬜ |

## J3 — Modèles (18 décembre)

| Tâche | Statut |
|---|---|
| GEN-07, GEN-08 : prototype de génération de problèmes sur `feature/generation-problemes`, intégration prévue en J3 | 🟨 |
| GEN-07 : dataset Q/R, taux de rejet, ≥ 5 % relus à la main | ⬜ |
| GEN-08 : LoRA / QLoRA sur le GPU CRIStAL, acceptation vs modèle de base | ⬜ |
| DET-04, DET-06, DET-07 : modèle français, AUROC hors distribution, FP ≤ 1 %, branché dans l'API | ⬜ |
| COR-05 : `poids_calibres.json` calibré et versionné, QWK ≥ 0,70 | ⬜ |
| Exigences sans jalon (ING-02/03/05, GEN-01/02/03, COR-02/03/04/06/07, DET-01/03/08) : mesurer | ⬜ |
| Rapport technique (en parallèle) | ⬜ |

## J4 — Interface complète (8 janvier)

| Tâche | Statut |
|---|---|
| GEN-04 : relire, modifier, supprimer, régénérer une question avant publication | ⬜ |
| WEB-03 : parcours bout en bout avec de vraies données | ⬜ |
| WEB-05, ENF-07, DET-05 : filtre par alerte, sous-scores, phrases suspectes surlignées | ⬜ |
| ENF-05 : accessibilité (skip link, focus, contrastes, Lighthouse, clavier) | ⬜ |
| WEB-04 : test avec 5 personnes chronométrées | ⬜ |
| ENF-06 : Chrome, Firefox, Safari, Edge, 360 px | ⬜ |
| Gel des fonctionnalités | ⬜ |

## J5 — Recette et soutenance (fin janvier)

| Tâche | Statut |
|---|---|
| ENF-04, 08, 09, 11, 12, 13 | ⬜ |
| Recette critère par critère, procès-verbal signé | ⬜ |
| Livrables §12 : rapport, OpenAPI, guide utilisateur, état de l'art, fiches datasets, démo | ⬜ |
| Répétition de soutenance | ⬜ |
| Fusion `develop` → `main`, tag `J5-recette` | ⬜ |

## Décisions

### Versionnement des modèles (CDC §8) — 7 nov. 2026

Les poids des modèles (`.pkl`, poids LoRA…) ne sont plus suivis par Git. Ils sont publiés
en **Release GitHub** (une release par modèle et par version, ex. `modele-detection-en-v1`,
`modele-detection-fr-v1`) et téléchargés à l'installation par
`scripts/telecharger_modeles.py`, qui vérifie leur empreinte SHA-256 (liste dans
`scripts/modeles.json`). Les petits fichiers lisibles (génomes, rapports d'entraînement,
`poids_calibres.json`) restent dans Git.

Pour publier un nouveau modèle : créer la release, y joindre le fichier, puis mettre à jour
`scripts/modeles.json` (URL + SHA-256) dans une PR.
