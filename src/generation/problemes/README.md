# Prototype — génération de sujets de mathématiques (pôle Génération)

Objectif : à partir d'un cours PDF, générer
- **A. des questions de cours** sur les notions importantes (définitions, théorèmes, méthodes) ;
- **B. un problème** en plusieurs questions progressives sur un thème du chapitre
  (ex. « intégrales de Wallis » pour un cours d'intégration), avec corrigé et barème
  **vérifiés automatiquement**.

Ce n'est pas encore branché à l'API ni au site : c'est un banc d'essai pour choisir les
modèles et mesurer la qualité (GEN-07, GEN-08) avant l'intégration.

## Le circuit

```
PDF ─► extraire  : chaque page rendue en image, transcrite en Markdown + LaTeX par un
                   modèle de vision (cache par page, fait une seule fois par cours)
    ─► analyser  : découpage par parties → fiches de notions (type, énoncé, importance 1-3)
    ─► probleme  : plan → rédaction (énoncé, corrigé, barème, vérifications déclarées)
                   → vérification numérique (SymPy/mpmath) des affirmations du corrigé
                   → résolution indépendante des questions de calcul, sans le corrigé
                   → correction ciblée des erreurs (--tours) → relecture critique
    ─► questions : questions de cours sur les notions d'importance 3
```

**Pourquoi la vision pour l'extraction :** dans les cours rédigés sous Word, la couche texte
du PDF rend les formules illisibles (sur le cours d'intégration testé, $I_n$ sort `𝐼'`,
$x^2$ sort `𝑥!` et le signe intégrale sort `Y`).

**Pourquoi la vérification numérique :** même un gros modèle se trompe dans un corrigé.
Le modèle déclare chaque résultat (valeur, récurrence, inégalité, limite) sous forme
d'expression ; `verification.py` les contrôle numériquement pour plusieurs valeurs de $n$.
Le modèle ne fournit jamais de code exécutable : les expressions passent par une liste
blanche de fonctions et de noms.

## Lancer sur un PC avec GPU (ex. RTX 2070, 8 Go)

```powershell
# 1. Modèles (une fois)
ollama pull qwen2.5vl:7b      # vision, pour l'extraction
ollama pull qwen3:8b          # génération avec raisonnement, tient en 8 Go
ollama pull gpt-oss:20b       # optionnel : à comparer (déborde un peu sur la RAM)

# Réglages conseillés pour une carte de 8 Go (à définir AVANT de lancer Ollama,
# puis redémarrer Ollama) : le cache d'attention prend 2 fois moins de mémoire
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0

# 2. Dépendances (depuis la racine du dépôt)
pip install -r requirements.txt

# 3. Circuit complet sur un cours
python -m src.generation.problemes tout "C:\chemin\integrales.pdf" --theme "intégrales de Wallis" --parties 4,5,6

# ou étape par étape
python -m src.generation.problemes extraire "C:\chemin\integrales.pdf"
python -m src.generation.problemes analyser sortie_problemes/cours.md
python -m src.generation.problemes probleme sortie_problemes/analyse.json --theme "intégrales de Wallis" --parties 4,5,6
python -m src.generation.problemes questions sortie_problemes/analyse.json --nb 5
```

Vérifier pendant une génération que tout tient sur la carte : `ollama ps` doit afficher
`100% GPU`. Sinon, réduire la fenêtre de contexte : `--num-ctx 12288`.

Options utiles : `--modele gpt-oss:20b --raisonnement medium` (gpt-oss utilise des niveaux),
`--raisonnement non` (plus rapide), `--difficulte 3`, `--tours 3`, `--pages 15-23` pour
n'extraire qu'une partie du cours, `--texte` pour sauter la vision (formules abîmées).

Les résultats arrivent dans `sortie_problemes/` (ignoré par Git : les cours peuvent être
protégés par le droit d'auteur, ne les commitez pas). Pour chaque problème :
`probleme_<thème>_<modèle>.md` (sujet, corrigé, rapport de vérification et durées) et `.json`.

## Comparer des modèles

Lancer `probleme` avec plusieurs `--modele` sur la même `analyse.json`, puis comparer dans
les rapports : vérifications ok / fausses, discordances de la résolution indépendante, note
de la relecture, durée — et surtout **votre jugement d'enseignant** sur le sujet.

## Limites connues

- La vérification numérique couvre les familles d'intégrales indexées par $n$ et les
  valeurs ; une démonstration rédigée n'est vérifiée qu'à travers les résultats qu'elle affirme.
- La relecture critique est faite par le même modèle : indicative seulement.
- Pas encore de banque d'exercices (RAG) : le modèle s'appuie sur ses connaissances du thème.
