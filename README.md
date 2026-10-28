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
git clone https://github.com/votre-org/testguardai.git
cd testguardai

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
│   │       └── main.py         # Backend FastAPI
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
| J1 — Intégration | ✅ Fait | WEB-01, COR-01, DET-05 |
| J2 — Sécurité et données | 1er nov. 2026 | WEB-02 ✅, DET-02 en cours |
| J3 — Modèles | 1er déc. 2026 | GEN-07, GEN-08, DET-04, COR-05 |
| J4 — Interface complète | 15 jan. 2027 | WEB-03→05, ENF-05 |
| J5 — Recette et soutenance | 31 jan. 2027 | Section 11 CDC |

## Branchements inter-pôles à faire (J3)

Les fonctions suivantes sont des stubs dans `main.py` — voir les `TODO J3` :

```python
# Pôle Génération (Anisse & Alexis)
# src/generation/generate.py
generer_questions(pdf_path: str, nb_questions: int, matiere: str) -> list[dict]

# Pôle Évaluation (Raphaël & Nina)
# src/evaluation/correct.py
corriger_reponse(reponse_etudiant: str, reponse_attendue: str) -> dict

# src/evaluation/detect.py
detecter_ia(texte: str) -> dict
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
