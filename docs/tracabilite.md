# Traçabilité exigences → code → tests

Une ligne par exigence du CDC : où elle est implémentée, comment elle est vérifiée.
Le libellé exact de chaque exigence fait foi dans le CDC (`docs/`).

| Exigence | Implémentation | Vérification | Statut |
|---|---|---|---|
| ING-01 — dépôt de PDF (≤ 20 Mo, contrôle du format) | `src/web/back/main.py` : `POST /documents` | `tests/test_j1_integration.py` | 🟨 |
| GEN — génération des questions depuis un cours | `src/generation/rag.py` : `ingest_and_generate_questions` ; `POST /devoirs/{id}/generer` | `tests/test_j1_integration.py` (modèle simulé) | 🟨 |
| COR-01 — correction automatique | `projet_ia.verification.evaluer_reponse` ; `POST /soumissions/{sid}/corriger-auto` | `tests/test_j1_integration.py` (modèle simulé) ; `src/evaluation/projet-ia/tests/` | 🟨 |
| DET — détection IA au dépôt et à la demande | `projet_ia.detection.predire_textes` ; `POST /soumissions`, `/soumissions/{sid}/analyser-ia` | `tests/test_j1_integration.py` (modèle simulé) ; `src/evaluation/projet-ia/tests/` | 🟨 |
| DET-05 — phrases suspectes visibles par l'enseignant | `src/web/index.html` : `surligner()` | à faire (test manuel avec captures) | 🟨 |
| ENF-10 — installation par `docker compose up` | `Dockerfile.api`, `docker-compose.yml` | à faire sur une machine vierge | ⬜ |

<!-- Compléter avec toutes les exigences du CDC (WEB, ING, GEN, COR, DET, ENF). -->
