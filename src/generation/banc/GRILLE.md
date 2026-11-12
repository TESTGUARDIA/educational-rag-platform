# Grille de notation enseignant — banc du pôle Génération

Chaque production (une question de cours, ou un problème complet) reçoit une note de
**0 à 2 sur 5 critères** (total sur 10), plus un avis « utilisable tel quel ».

Les productions sont **anonymisées** : le code (C001, C002…) ne dit pas quel modèle les a
produites. N'ouvrez `correspondance.json` qu'**après** avoir tout noté.

| Critère (colonne) | 0 | 1 | 2 |
|---|---|---|---|
| `exactitude` | Erreur mathématique ou factuelle (énoncé faux, corrigé faux) | Imprécision mineure, notation maladroite | Exact |
| `conformite_programme` | Hors du cours ou du programme officiel | À la limite (notion voisine, prérequis lourd) | Conforme au cours et au programme |
| `interet` | Porte sur un détail anodin, ou trivial | Utile mais secondaire | Porte sur une notion centrale, fait réfléchir |
| `difficulte_adaptee` | Inadaptée au niveau (bien trop facile ou bien trop dure) | Un peu trop facile ou trop dure | Adaptée au niveau visé |
| `clarte` | Ambigu, incompréhensible sans le corrigé | Compréhensible avec effort | Clair et autonome |

`utilisable` : **oui** (je le donnerais tel quel), **modif** (utilisable après retouches
mineures), **non** (à jeter). Le taux de « oui » + « modif » est le **taux d'acceptation**,
et le taux de « non » le **taux de rejet** (GEN-07).

`commentaire` : facultatif, mais très utile pour améliorer les prompts.

Pour un **problème**, notez l'ensemble (énoncé + corrigé + progression). Pour le comparer à
une référence humaine, voir le champ « référence humaine » du cas dans `a_noter.md`.

Remplissez `notation.csv` (séparateur « ; », s'ouvre dans Excel ou LibreOffice), puis :

    python -m src.generation.banc resumer sortie_banc/<dossier du lancement>
