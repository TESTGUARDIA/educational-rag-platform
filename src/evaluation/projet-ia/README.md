# projet-ia

Deux outils de traitement automatique du français, utilisables en Python ou en ligne de commande :

| Outil | Rôle |
|---|---|
| **Vérification** (`projet_ia.verification`) | Évalue si la réponse d'un élève est correcte par rapport à une correction (calcul numérique/symbolique, NLI, embeddings CamemBERT). |
| **Calibration** (`projet_ia.calibration`) | Entraîne (régression logistique, descente de gradient) les poids qui combinent les sous-scores de la vérification. |
| **Détection** (`projet_ia.detection`) | Détecte si un **paragraphe** (français ou anglais) a été écrit par une IA ou un humain : trois analyses (paragraphe, phrase par phrase, comparaison entre phrases), chacune par un MLP optimisé par algorithme génétique, fusionnées par une somme pondérée de log-odds dont les poids sont appris par descente de gradient. |

Python **3.10 ou plus** requis.

## Installation

```bash
git clone <url-du-depot>
cd projet-ia

python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

pip install -e ".[dev]"            # installe le projet et ses dépendances
python -m spacy download fr_core_news_md
python -m spacy download fr_core_news_sm
```

Le premier lancement télécharge CamemBERT et le modèle NLI DeBERTa (plusieurs Go) ; ils sont ensuite en cache.

## Organisation du dépôt

```
projet-ia/
├── projet_ia/
│   ├── config.py            # chemins des fichiers et noms des modèles (tout se règle ici)
│   ├── cli.py               # ligne de commande : python -m projet_ia ...
│   ├── common/              # chargement des modèles (paresseux), embeddings
│   ├── verification/        # evaluer_reponse() et ses briques
│   │   ├── evaluateur.py    #   point d'entrée : evaluer_reponse()
│   │   ├── question.py      #   analyse de la question
│   │   ├── numerique.py     #   pipeline numérique (sympy)
│   │   ├── semantique.py    #   pipeline sémantique (NLI + embeddings)
│   │   ├── poids.py         #   poids calibrés et score_calibre()
│   │   └── interactif.py    #   mode interactif et démo
│   ├── calibration/         # entraînement des poids (calibrer())
│   └── detection/
│       ├── donnees.py       #   lecture du dataset (parquet / jsonl / csv), filtre de langue
│       ├── langue.py        #   détection automatique français / anglais
│       ├── style.py         #   features de style, de phrase, de comparaison entre phrases
│       ├── cache.py         #   pré-calcul unique des embeddings de phrases (reprenable)
│       ├── composants.py    #   les 3 analyses : paragraphe / phrase / interphrases
│       ├── mlp.py           #   GenomeMLP, entraînement par époque, métriques
│       ├── evolution.py     #   algorithme génétique (n'ouvre jamais le jeu de test)
│       ├── fusion.py        #   somme pondérée de log-odds (descente de gradient)
│       ├── entrainement.py  #   entraînement final + fusion + test + rapport
│       ├── prediction.py    #   prédire IA / humain (langue automatique)
│       ├── grammaire.py     #   outils d'analyse grammaticale (spaCy, fr / en)
│       └── relations.py     #   outils d'analyse des relations syntaxiques (français)
├── data/                    # vos CSV (non versionnés)
├── models/                  # fichiers générés : poids, génomes, .pkl (non versionnés)
└── tests/
```

## Utilisation

### 1. Vérifier une réponse

```bash
python -m projet_ia verifier                      # mode interactif
python -m projet_ia verifier --demo               # exemples de démonstration
python -m projet_ia verifier --question "Qui a écrit Hamlet ?" \
       --reponse "Victor Hugo" --correction "William Shakespeare"
```

```python
from projet_ia.verification import evaluer_reponse

r = evaluer_reponse(
    question   = "Quelle est la capitale de la France ?",
    reponse    = "C'est Paris.",
    correction = "Paris est la capitale de la France.",
)
print(r["score"], r["verdict"])        # ex. 0.87 Correct
```

### 2. Calibrer les poids de la vérification

```bash
python -m projet_ia calibrer                                   # data/meta_dataset.csv -> models/poids_calibres.json
python -m projet_ia calibrer --dataset data/mon_fichier.csv --epochs 8000 --lr 0.05
```

```python
from projet_ia.calibration import calibrer
calibrer("data/meta_dataset.csv", epochs=5000)
```

`data/meta_dataset.csv` doit contenir les colonnes : `label` (0/1), `caractere_question`
(`fermée` / `semi-ouverte` / `ouverte`), `nli`, `similarite_ponderee`, `sens_global`,
`penalite_neg`, `penalite_entites`, `penalite_concept`.

Si `models/poids_calibres.json` n'existe pas, la vérification utilise des poids de secours
(un message l'indique). Une fois généré, **commitez ce fichier** : c'est lui qui donne les
mêmes résultats à toute l'équipe.

### 3. Détection IA / humain (français ou anglais)

**Principe.** Chaque paragraphe est découpé en phrases ; chaque phrase est encodée **une seule
fois** par un Transformer, et le paragraphe passe une fois dans un **modèle de langue causal**
(`gpt2` en anglais, `bigscience/bloom-560m` en français) pour mesurer sa **perplexité** : à quel
point chaque token était prévisible. Un LLM choisit des tokens probables, donc son texte est peu
surprenant, et régulièrement d'une phrase à l'autre. Ce modèle de langue n'est **jamais entraîné** :
c'est un instrument de mesure (`detection/perplexite.py`). Trois analyses, chacune avec son propre
MLP, en sont déduites :

| Analyse | Ce qu'elle regarde |
|---|---|
| `paragraphe` | embedding moyen des phrases + style global + cohérence + perplexité |
| `phrase` | chaque phrase isolément, perplexité comprise (les phrases héritent du label du paragraphe) |
| `interphrases` | comparaison des phrases entre elles : régularité de la longueur, de la ponctuation, du vocabulaire, de la perplexité, similarité sémantique entre phrases successives |

Features de perplexité : log-probabilité moyenne des tokens, son écart-type, part des tokens qui
étaient le choix n°1 du modèle, entropie moyenne, rapport surprise / entropie (corrige les textes
prévisibles par nature) et écart de perplexité entre phrases. L'algorithme génétique peut activer
ou non ce bloc (`use_ppl`) pour chaque analyse.

Leurs sorties sont fusionnées : `score = biais + Σ poids_ia·z⁺ + Σ poids_humain·z⁻`, où `z` est le
log-odds de chaque analyse (pour les phrases : moyenne, maximum et minimum). Les poids « IA » et
« humain » sont **distincts** et appris par **descente de gradient** (`calibration/regression.py`).

**Langues.** `--langue en|fr|multi`. Chaque langue a son encodeur (`roberta-base`,
`camembert-base`, `xlm-roberta-base`), son cache (`data/cache/<langue>`) et ses modèles
(`models/<langue>/`), tous réglables dans `projet_ia/config.py`. À la prédiction, la langue est
détectée automatiquement (`--langue auto`, par défaut) ; sans modèle pour la langue détectée, le
modèle `multi` sert de repli. Un modèle entraîné sur de l'anglais ne sait pas analyser du français :
il faut un dataset français pour entraîner `fr`.

**Dataset** : un fichier `.parquet`, `.jsonl` ou `.csv` (par défaut `data/dataset.parquet`) avec au
minimum un texte, un label (`ai` / `human`) et un `split` (`train` / `val` / `test`). Les colonnes
`domain`, `source`, `model`, `length_category` sont facultatives (utilisées pour le rapport) ; une
colonne `lang` permet de filtrer par langue. Les noms acceptés sont dans `config.COLONNES_DATASET`.
Le `split` doit être construit **par groupes** (un texte humain et ses réécritures IA dans le même
split), sinon le score de test est gonflé.

```bash
# 1. Pré-calcul des features : étape LENTE, faite une seule fois par langue, reprenable
python -m projet_ia detection preparer --langue en --n-jobs 4
#    essai rapide d'abord :  --max-train 20000 --max-val 5000 --max-test 5000

# 2. Recherche du meilleur MLP de chaque analyse (génétique), puis fusion, test et rapport
python -m projet_ia detection genetique --langue en
#    essai rapide :  --population 10 --elite 4 --generations 5 --lot-train 10000
#    une seule analyse :  --composants interphrases

# 3. Prédiction (langue détectée automatiquement)
python -m projet_ia detection predire "Un paragraphe de plusieurs phrases. Le modèle compare leur style."
```

L'étape 2 produit dans `models/<langue>/` : `modele_detection.pkl`, un `genome_<analyse>.json` et un
`historique_evolution_<analyse>.json` par analyse, et **`rapport_entrainement.json`** (métriques de
validation et de test de chaque analyse seule ET de la fusion, poids de fusion, courbes de perte,
accuracy par domaine / longueur, taux de détection par modèle générateur). Pour refaire seulement
l'entraînement final avec les génomes déjà trouvés : `python -m projet_ia detection entrainer`.

**Seuil de décision.** Accuser un élève à tort coûte plus cher que laisser passer un texte IA : le
seuil n'est donc pas 0.5, mais le plus bas seuil qui classe « IA » au plus **1 %** des textes humains
de `val` (`config.FPR_CIBLE_DETECTION`, ou `--fpr-cible 0.005` à l'entraînement). Il est enregistré
dans `modele_detection.pkl` et utilisé par défaut par `predire` (`--seuil` pour le forcer). Le rapport
donne, sur `test`, le taux de faux positifs réellement obtenu et la part d'IA détectées à ce seuil
(`test`), à comparer avec le seuil 0.5 (`test_seuil_0_5`).

```python
from projet_ia.detection import (charger_dataset, construire_cache, algorithme_genetique,
                                 entrainer_final, predire_depuis_fichier, COMPOSANTS)

df = charger_dataset("data/dataset.parquet", langue="en")
construire_cache(df, langue="en", n_jobs=4)
genomes = {c: algorithme_genetique("data/cache/en", composant=c) for c in COMPOSANTS}
entrainer_final(genomes, "data/cache/en")
r = predire_depuis_fichier("Un paragraphe à tester. Avec plusieurs phrases.")
print(r["prediction"], r["probabilite_ia"], r["composants"], r["phrases"])
```

Le résultat détaille la probabilité de chaque analyse et de **chaque phrase** (utile pour voir quelles
phrases paraissent générées).

**Rigueur de l'évaluation** : la recherche génétique ne lit que `train` et `val` ; le split `test`
n'est ouvert qu'**une seule fois**, à la fin de `entrainer_final`. Les poids de fusion sont appris
sur `val`, avec des prédictions que les analyses n'ont pas vues à l'entraînement. Ne choisissez pas
l'encodeur ou les génomes en regardant le score de test : il ne serait plus un test.

Un texte de moins de 3 phrases déclenche un avertissement de fiabilité (la comparaison entre
phrases n'a alors presque rien à comparer).

**Modèle anglais fourni** (`models/en/modele_detection.pkl`, versionné : `predire` fonctionne sans
réentraîner). Entraîné sur 1,22 million de textes anglais (`dataset5.parquet` : RAID, MAGE, M4GT,
HC3…), évalué une seule fois sur 67 903 textes de test :

| Au seuil appris (0,80) | Valeur |
|---|---|
| Textes humains classés IA (faux positifs) | 0,99 % |
| Textes IA détectés | 91,3 % |
| Précision / accuracy / AUC | 98,9 % / 95,2 % / 0,994 |

Textes courts : 91,8 % d'accuracy (96–97 % pour les moyens et longs). Générateurs les moins bien
détectés : Dolly, FLAN-T5, davinci (67–76 %). **Anglais uniquement** : il ne doit pas servir sur du
français. Détail complet (par domaine, longueur, générateur, poids de fusion) dans
`models/en/rapport_entrainement.json`.

## Tests

```bash
pytest                 # rapide : n'a besoin d'aucun modèle (encodeur simulé)
pytest -m modeles      # lent : bout en bout avec les vrais modèles
```

## Contribuer

* Chemins et noms de modèles : **uniquement** dans `projet_ia/config.py`.
* Les modèles lourds ne sont jamais chargés à l'import : utiliser `common/models.py`.
* Ne pas commiter `data/` ni `models/` (sauf `models/poids_calibres.json`).
