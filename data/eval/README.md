# Jeu d'évaluation — règles d'annotation

Le fichier [`questions.jsonl`](questions.jsonl) est l'étalon de mesure de tout le
projet. Les étapes 12 à 30 sont des comparaisons ; une comparaison a besoin d'une
règle graduée. Ces règles existent pour qu'une question annotée dans six mois le
soit comme celles d'aujourd'hui.

## Format

Une question par ligne, en JSON. Les lignes vides et celles commençant par `#`
sont ignorées, donc le fichier peut être annoté par sections.

```json
{"question_id": "q001", "question": "...", "category": "conceptual", "relevant_document_ids": ["fastapi:tutorial/dependencies/index"], "notes": "facultatif", "held_out": false}
```

| Champ | Rôle |
|---|---|
| `question_id` | `qNNN`, unique, jamais réattribué même si la question est supprimée |
| `question` | la question telle qu'un utilisateur l'écrirait |
| `category` | `conceptual`, `exact`, `code`, `multi_doc` ou `unanswerable` |
| `relevant_document_ids` | la vérité terrain, au niveau **document** |
| `relevant_sections` | facultatif, pour l'étape 12 si « bonne page, mauvaise section » compte |
| `notes` | pourquoi cette annotation, si ce n'est pas évident |
| `held_out` | mis de côté ; on ne le regarde que quand un résultat semble trop beau |

## Annotation au niveau document, jamais au niveau chunk

Les `chunk_id` changent à chaque modification de `chunk_size` : une vérité terrain
au niveau chunk devrait être refaite pour chacune des cinq stratégies comparées à
l'étape 12. Personne ne réannote quarante questions cinq fois, donc la comparaison
n'aurait tout simplement jamais lieu. Les `document_id` (étape 02) survivent au
re-découpage, au re-nettoyage et au re-clonage.

Le coût est réel et assumé : on ne distingue pas « a trouvé la bonne page, mauvaise
section » de « a trouvé la bonne section ». `relevant_sections` est là pour les
rares questions où cette distinction changera une décision.

Un chunk retrouvé compte comme une réussite quand son `document_id` figure dans
`relevant_document_ids`.

## Écrire la question *à partir* du corpus

Ouvrir un document, y chercher ce qu'une vraie personne demanderait, puis écrire la
question **avec ses mots à elle**, pas avec ceux du document. Une question qui
reprend la formulation exacte du document teste la correspondance de chaînes, pas
la recherche, et fait paraître la recherche dense meilleure qu'elle n'est.

Ensuite — et c'est l'étape qu'on saute quand on est pressé — chercher dans le corpus
les *autres* documents qui répondent aussi, et ajouter leurs identifiants. Oublier
un document réellement pertinent pénalise un système qui avait raison : c'est ainsi
qu'une bonne amélioration se fait rejeter par un mauvais étalon.

## Les cinq catégories

| Catégorie | Ce qu'elle sonde | Minimum |
|---|---|---|
| `conceptual` | terrain de jeu de la recherche dense | 8 |
| `exact` | identifiants, codes d'erreur, noms de classes — le cas BM25 (étape 14) | 8 |
| `code` | récupération de blocs de code | 8 |
| `multi_doc` | questions couvrant 2 à 4 documents | 8 |
| `unanswerable` | comportement de refus (étape 22) — `relevant_document_ids: []` | 8 |

Sans `exact` ni `unanswerable`, les étapes 14 et 22 n'ont rien à démontrer et leurs
chiffres seraient inventés.

Les questions `exact` s'écrivent comme on les taperait : l'identifiant, sans prose
autour. Les questions `unanswerable` doivent rester *proches* du corpus — un autre
framework web, un outil voisin. « Comment faire du pain » ne teste rien.

## Taille

40 à 50 questions, au moins 8 par catégorie. En dessous de ~30, une seule question
fait bouger le Recall@5 plus que les améliorations mesurées. Au-dessus de ~60, le
jeu n'est jamais terminé — et un jeu inachevé vaut zéro question.

Le nombre de documents pertinents doit **varier** d'une question à l'autre. Si
chaque question a exactement une réponse, le Recall@K se confond avec le Hit Rate
et deux des cinq métriques de l'étape 11 deviennent redondantes.

## Le jeu réservé (`held_out`)

Cinq questions environ, réparties sur les catégories. On ne les regarde jamais en
ajustant la recherche. Régler le système en examinant les échecs individuels du jeu
principal, c'est du surapprentissage sur 45 questions ; le jeu réservé est le seul
moyen de s'en apercevoir.

## Vérifier

```powershell
uv run python scripts/validate_dataset.py
```

Le script recoupe chaque `relevant_document_id` avec le corpus nettoyé. Un
identifiant absent — faute de frappe, ou document écarté par le filtre de l'étape
03 — rend la question définitivement sans réponse tout en ayant l'air correcte.
