# TestGuardAI — Plateforme d'évaluation académique avec détection IA

Plateforme confidentielle pour la génération de tests, la correction automatique
et la détection de l'usage d'une IA dans les copies étudiantes.

## Stack technique

| Composant | Technologie |
|---|---|
| Frontend | HTML / JS vanilla — servi par Nginx |
| Backend | Python 3.11 + FastAPI + SQLite |
| Auth | JWT (RFC 9068) + OAuth 2.1 PKCE (RFC 9700) |
| Déploiement | Docker Compose |

## Démarrage rapide

### Prérequis
- Docker ≥ 24
- Docker Compose ≥ 2.20

### Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/TESTGUARDIA/educational-rag-platform.git
cd educational-rag-platform

# 2. Créer le fichier d'environnement
cp .env.example .env

# 3. Générer une clé secrète JWT et l'ajouter dans .env
python -c "import secrets; print(secrets.token_hex(32))"
# → Copier le résultat dans JWT_SECRET= dans .env

# 4. Lancer
docker compose up -d --build
```

Le site est disponible sur **http://localhost**.

### Comptes de démonstration

| Rôle | Email | Mot de passe |
|---|---|---|
| Enseignant | marie.dupont@testguard.ai | password |
| Étudiant | lucas.moreau@testguard.ai | password |

> ⚠️ Changer les mots de passe avant tout déploiement non local.

## Structure du projet

```
testguardai/
├── src/
│   ├── web/
│   │   ├── index.html          # Frontend complet (HTML/JS vanilla)
│   │   └── back/
│   │       ├── main.py         # Backend FastAPI
│   │       └── services.py     # Branchements vers les pôles Génération et Évaluation
│   ├── data/                   # Datasets (voir pôle Données)
│   └── evaluation/             # Modules de correction/détection (voir pôle Évaluation)
├── Dockerfile.api              # Image Docker backend
├── Dockerfile.frontend         # Image Docker frontend (Nginx)
├── docker-compose.yml          # Orchestration
├── nginx.conf                  # Configuration Nginx + proxy API
├── requirements.txt            # Dépendances Python
└── .env.example                # Variables d'environnement (modèle)
```

## Jalons

| Jalon | Date | État |
|---|---|---|
| J1 — Intégration | juin 2026, rattrapage 13 nov. 2026 | Atteint dans `develop` : génération, correction, détection appelables par l'API (WEB-01, COR-01, DET-05) |
| J2 — Sécurité et données | 30 nov. 2026 | WEB-02 fait (JWT, rôles) ; restent le rôle administrateur, DET-02, ENF-02, ENF-03 |
| J3 — Modèles | 18 déc. 2026 | GEN-07, GEN-08, DET-04, DET-06, DET-07, COR-05 |
| J4 — Interface complète | 8 janv. 2027 | GEN-04, WEB-03 à 05, ENF-05 à 07, DET-05 |
| J5 — Recette et soutenance | fin janv. 2027 | Recette §11, livrables §12 |

Détail et état de chaque exigence : [docs/avancement.md](docs/avancement.md).

## Organisation du dépôt

- `main` ne reçoit `develop` qu'à chaque jalon, avec un tag (`J1-integration`, …).
- `develop` est la branche d'intégration : on y entre par Pull Request, relue par
  un membre d'un autre pôle (CDC §9).
- Chaque pôle travaille sur une branche `feature/...` créée depuis `develop`
  (`feature/charte-graphique`, `feature/generation-problemes`, …). Les anciennes branches
  sont conservées en tags `archive/...`.
- Les poids des modèles ne sont pas dans Git (CDC §8) : ils sont publiés en Release
  GitHub et téléchargés à l'installation.

## Branchements inter-pôles (J1)

Le backend web (`src/web/back/main.py`) appelle les pôles uniquement via
[`src/web/back/services.py`](src/web/back/services.py) :

| Fonction (`services.py`) | Pôle | Code appelé | Routes |
|---|---|---|---|
| `generer_questions(pdf, nom, nb_questions)` | Génération | `src/generation/rag.py` → `ingest_and_generate_questions` (Ollama) | `POST /devoirs/{id}/generer` |
| `corriger_reponse(question, reponse, correction)` | Évaluation | `projet_ia.verification.evaluer_reponse` | `POST /soumissions/{sid}/corriger-auto` |
| `detecter_ia(textes)` | Évaluation | `projet_ia.detection.predire_textes` | `POST /soumissions` (arrière-plan), `POST /soumissions/{sid}/analyser-ia` |

- La détection utilise le **seuil appris par le modèle** (faux positifs bornés à 1 %),
  repris par `/stats` et les badges. Sans modèle pour la langue de la copie (le français
  pour l'instant, DET-04 à J3), le rapport est `non_analysé`.
- La note de la correction automatique est une **suggestion** : l'enseignant la valide
  avec `POST /soumissions/{sid}/noter`.

### Prérequis du premier lancement

- **Ollama sur la machine hôte**, avec le modèle de `.env` (`OLLAMA_MODEL`) :
  `ollama pull qwen2.5:7b`. Sans GPU NVIDIA, un modèle 7B demande plusieurs minutes
  par question : préférer un modèle plus petit pour une démonstration.
- **≈ 3 Go de modèles Hugging Face** (CamemBERT, DeBERTa, RoBERTa, GPT-2) téléchargés
  au premier démarrage dans le volume `testguardai_hf_cache` (`PRECHARGER_MODELES=1`).

### Tests

```bash
pip install -r requirements.txt pytest
pytest -q                     # backend web + branchements (modèles simulés)
```

## Équipe

- **Éloïse Guimbellot** — Cheffe de projet, Pôle Web
- **Thomas Tanneur** — Pôle Web
- **Raphaël Johnson** — Pôle Évaluation
- **Nina Schlienger** — Pôle Évaluation
- **Anisse Benalegue** — Pôle Génération
- **Alexis Doussaud-Pelletier** — Pôle Génération
- **Arsène Brégeon** — Pôle Données
- **Juliette Bros** — Pôle Données
- **Virna Martinez** — Intégration (Web, Génération)
