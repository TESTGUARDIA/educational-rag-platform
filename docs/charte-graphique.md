# Charte graphique TestGuardAI

Appliquée au site vanilla (`src/web/index.html`) et à la version React (`src/web/react/`).

## Logo
- Symbole : le T en chapeau de diplômé (pompon et bouton rouge correcteur), le G en loupe (violet IA).
- Nom en **Bricolage Grotesque ExtraBold**, police réservée au logo. « AI » dans une pastille violette.
- Versions : empilée (connexion, grands formats), en ligne (en-tête), favicon (`docs/favicon.svg`).
- Ne jamais recolorer le symbole ni poser le logo sur une couleur de matière.

## Couleurs
| Rôle | Valeur | Usage |
|---|---|---|
| Violet IA | `#6D4AE0` (survol `#5638C4`, teinte `#ECE6FC`) | Boutons, liens, champ actif, loupe, pastille AI |
| Rouge correcteur | `#D83A2E` graphique · `#B92E24` texte · teinte `#FAE6E0` | Logo et annotations de correction **uniquement** |
| Noir | `#1C1B22` | Tout le texte, alerte forte |
| Gris | `#55535E` | Texte secondaire |
| Ivoire | `#FAF7F1` | Fond de page |
| Page | `#FFFDF8` | Panneaux, tableaux, fenêtres |
| Bordure | `#E3DDD1` | Contours 1 px |

## Score IA
Le niveau suit le seuil renvoyé par le modèle (0,798 pour le modèle anglais) et monte en poids
visuel, jamais en couleur : Faible (sous 50 %, gris) · À vérifier (de 50 % au seuil, contour noir)
· Alerte (au-dessus du seuil, noir plein). Les copies non analysées (langue sans modèle) restent
marquées « non analysé ».

## Matières
Dix couleurs choisies par l'enseignant : Mandarine, Soleil, Pomme, Émeraude, Sarcelle, Lagon, Océan,
Corail, Terre, Ardoise. Rouges et violets exclus (réservés à la marque). Initiales sur la tuile,
texte blanc ou noir calculé pour rester lisible. Une couleur par matière.
Vanilla : en place dans « Ajouter une matière ». React : palette prête dans `src/utils/matieres.js`.

## Typographie et règles
- Site : **Montserrat** 700 (titres), 600 (intertitres, boutons), 400–500 (texte). Chiffres à chasse fixe.
- Rayon 8 px (blocs), 6 px (boutons, champs, étiquettes) · bordures 1 px.
- Aucune ombre, aucun dégradé, aucun emoji · cibles cliquables ≥ 40 px · focus visible violet.
- Enseignant : dense et neutre. Élève : plus aéré, couleurs de matières visibles.
