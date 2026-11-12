# Banc de mesure du pôle Génération (phase 0)

Objectif : **choisir le modèle et la méthode sur des chiffres**, et produire la preuve
attendue pour GEN-07 (taux d'acceptation / de rejet) et GEN-08 (gain d'un fine-tuning).

## En 4 commandes

```powershell
python scripts/telecharger_cours_test.py        # 1. cours de test (une fois)
python -m src.generation.banc liste             # 2. cas et configurations
python -m src.generation.banc lancer            # 3. génère tout (plusieurs heures avec 4 configs)
python -m src.generation.banc resumer sortie_banc/<date>   # 4. après la notation
```

Entre 3 et 4 : un enseignant (toi, Alexis, ou mieux un prof de maths) lit
`sortie_banc/<date>/a_noter.md` et remplit `notation.csv` selon [GRILLE.md](GRILLE.md).
**Les productions sont anonymisées** (C001, C002…) : on note sans savoir quel modèle a
produit quoi. `correspondance.json` ne s'ouvre qu'après.

`resumer` produit `synthese.md` et `synthese.csv` : par configuration, la note enseignant
sur 10, le taux d'acceptation et de rejet, et les mesures automatiques (vérifications du
corrigé réussies, erreurs restantes, doublons, durées).

## Lancer seulement une partie

```powershell
python -m src.generation.banc lancer --cas maths_integrales_wallis --configs qwen3-8b-raisonnement,gpt-oss-20b
python -m src.generation.banc lancer --configs llama3.2-3b-essai --cas svt_photosynthese_questions   # essai rapide
```

Les extractions et analyses sont en cache (`sortie_banc/cache/`) : relancer une autre
configuration sur le même cours ne refait que la génération.

## Fichiers

- [cas.json](cas.json) : cours, mode (`questions` ou `probleme`), thème, pages, matière.
  Le cas `maths_monka_wallis` utilise le cours d'Yvan Monka (droits réservés) : donner son
  chemin dans la variable `TESTGUARDAI_COURS_MONKA`, il n'est jamais commité.
- [configs.json](configs.json) : modèle Ollama, raisonnement, extraction (`vision` ou `texte`).
  Ajouter une configuration = ajouter une entrée (`ollama pull` du modèle avant).
- [GRILLE.md](GRILLE.md) : critères de notation.

`sortie_banc/` est ignoré par Git (il contient des extraits de cours).
