# Lancer le banc sur un PC avec carte NVIDIA (consignes pour Claude Code)

Ce fichier s'adresse à Claude Code, ouvert sur le PC où le banc doit tourner (prévu : portable
avec RTX 2070). Suis les étapes dans l'ordre. **Ne saute pas l'étape 1** : les choix de l'étape 4
dépendent des caractéristiques réelles de la machine, pas de celles annoncées.

Contexte : le banc (`python -m src.generation.banc`) génère des questions de cours et des
problèmes à partir de cours PDF avec plusieurs configurations de modèles Ollama, puis prépare
une notation **à l'aveugle** par un humain. Voir [README.md](README.md) et [GRILLE.md](GRILLE.md).

## Règles

- Ne modifie pas `.env`. Ne change aucun réglage Windows (alimentation, variables
  d'environnement, veille) **sans l'accord de l'utilisateur** : propose la commande, attends le oui.
- Ne commite jamais `cours_test/`, `sortie_banc/`, `sortie_problemes/` ni aucun PDF.
  Aucun commit ni push sans demande explicite ; jamais sur `main` ou `develop`.
- **Ne note pas les productions.** N'ouvre pas `correspondance.json`, ne remplis pas
  `notation.csv`, ne donne pas d'avis sur la qualité des sorties de `a_noter.md` : la notation
  est faite par un humain qui ne sait pas quel modèle a produit quoi. Tu peux rapporter les
  mesures automatiques (statut, durées, vérifications, erreurs).
- Si une étape échoue, diagnostique avant de contourner (journal Ollama :
  `$env:LOCALAPPDATA\Ollama\server.log`).

## 1. Vérifier les caractéristiques de la machine

Lance ces commandes (PowerShell) et présente le résultat à l'utilisateur sous forme de tableau :

```powershell
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv   # carte, mémoire vidéo, pilote
Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores
"{0:N0} Go de RAM" -f ((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
Get-PSDrive C | Select-Object @{n='Libre (Go)';e={[math]::Round($_.Free/1GB)}}
(Get-CimInstance Win32_Battery).BatteryStatus    # 2 = sur secteur ; vide = PC fixe
powercfg /getactivescheme
python --version
ollama --version
[Environment]::GetEnvironmentVariable('OLLAMA_FLASH_ATTENTION','User')
[Environment]::GetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE','User')
```

Puis conclus selon ce tableau :

| Constat | Conséquence |
|---|---|
| Pas de carte NVIDIA ou `nvidia-smi` introuvable | Arrête-toi : signale-le (pilote à installer, ou mauvais PC). |
| Mémoire vidéo ≥ 8 Go | Configurations par défaut. `qwen3:8b` tient presque entièrement sur la carte. |
| Mémoire vidéo de 6 Go | Possible mais 2 fois plus lent environ ; préviens l'utilisateur avant de continuer. |
| RAM < 24 Go | Exclure `gpt-oss-20b` (il déborderait sur le disque). Avec 32 Go, le garder. |
| Disque C : < 35 Go libres | Préviens : les modèles pèsent ~25 Go au total. |
| Portable sur batterie (`BatteryStatus` ≠ 2) | Demande de brancher le secteur : sur batterie la carte ralentit fortement. |
| Mode d'alimentation autre que « Performances » | Propose de le changer (avec accord). |
| Python < 3.10 ou absent | Faire installer Python 3.11 ou 3.12 (le code utilise `X | None`). |
| Ollama absent | Faire installer Ollama depuis ollama.com. |

Rappelle aussi à l'utilisateur, pour un portable : laisser l'écran ouvert, aérations dégagées,
mise en veille désactivée sur secteur pendant le banc (Paramètres → Système → Alimentation).

## 2. Préparer le dépôt et Python

Depuis la racine du dépôt, sur la branche `feature/generation-problemes` :

```powershell
git fetch origin
git switch feature/generation-problemes
git pull
python -m venv .venv            # si .venv n'existe pas
.venv\Scripts\Activate.ps1
pip install -r requirements.txt pytest
python -m pytest tests/test_banc.py tests/test_problemes_verification.py tests/test_problemes_generation.py tests/test_problemes_decoupage.py -q
python scripts/telecharger_cours_test.py
python -m src.generation.banc liste
```

Les tests doivent passer et `liste` doit montrer les cas avec `[ok]`. Le cas
`maths_monka_wallis` reste `[PDF absent]` sauf si l'utilisateur donne le chemin de ce cours
dans `TESTGUARDAI_COURS_MONKA` (cours sous droits réservés, jamais dans le dépôt) : c'est normal.

## 3. Préparer Ollama

Si les deux variables de l'étape 1 sont vides, propose (avec accord) :

```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
```

Elles divisent par 2 la mémoire du cache d'attention. Ollama doit ensuite être **quitté**
(icône près de l'horloge → Quit) puis relancé, sinon elles ne sont pas prises en compte.

Télécharge les modèles (~25 Go, à faire une fois) :

```powershell
ollama pull qwen3:8b
ollama pull qwen2.5vl:7b
ollama pull gpt-oss:20b      # seulement si gardé à l'étape 1
```

## 4. Essai court (≈ 5 à 10 min)

```powershell
python -m src.generation.banc lancer --cas svt_photosynthese_questions --configs qwen3-8b-sans-raisonnement
```

Pendant qu'il tourne, dans un autre terminal : `ollama ps`. La colonne PROCESSOR doit afficher
`100% GPU` ou presque. Si une part importante est sur le CPU : cherche ce qui occupe la carte
(`nvidia-smi`, navigateur, autre modèle chargé), et vérifie que les variables de l'étape 3 sont
actives. Donne à l'utilisateur la durée de l'essai et le statut obtenu.

## 5. Banc complet (plusieurs heures)

Avec 32 Go de RAM et `gpt-oss:20b` gardé :

```powershell
python -m src.generation.banc lancer
```

Sinon, sans lui :

```powershell
python -m src.generation.banc lancer --configs qwen3-8b-raisonnement,qwen3-8b-sans-raisonnement,qwen3-8b-texte-brut
```

Lance-le en arrière-plan et surveille-le. La première fois, l'extraction par vision des cours de
maths prend du temps ; elle est mise en cache (`sortie_banc/cache/`) et n'est pas refaite. Si le
banc s'arrête en cours de route, relance la même commande : les extractions et analyses déjà
faites sont reprises du cache.

## 6. Compte rendu à l'utilisateur

Donne :
- le tableau des caractéristiques (étape 1) et les configurations lancées ;
- la durée totale, et pour chaque cas × configuration : statut, durée, erreurs éventuelles
  (mesures automatiques uniquement, pas d'avis sur la qualité) ;
- le chemin du dossier `sortie_banc/<date>` et la suite :
  1. un enseignant lit `a_noter.md` et remplit `notation.csv` selon `GRILLE.md`
     (fichier CSV séparé par `;`, s'ouvre dans Excel) ;
  2. puis `python -m src.generation.banc resumer sortie_banc/<date>` produit `synthese.md`.
