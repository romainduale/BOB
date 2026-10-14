# Réponses au workbook — Day 1 et Day 2

Ce document répond aux questions de `docs/DAY1.md` et `docs/DAY2.md` à
partir des données officielles locales du projet. Les scores ont été recalculés
avec les scripts et dépendances actuellement présents dans le dépôt.

## Repères de validation

- **Holdout aléatoire 80/20** : utile pour les premiers contrôles, mais des
  visites d'un même patient peuvent se retrouver des deux côtés.
- **GroupKFold à 5 folds par `patient_id`** : estimation à privilégier, car un
  patient est entièrement dans le train ou dans la validation.
- Les valeurs recalculées pour le Day 2 sont : Dummy `16,50`, Ridge `10,60`,
  HGBR numérique `7,76` et TableVectorizer + HGBR `7,53`.
- Le fichier OOF historique de l'analyse d'erreurs donne `7,54`. L'écart de
  `0,01` avec le recalcul (`7,53`) est négligeable et peut venir des versions de
  bibliothèques ou du run sauvegardé.

# Day 1

## 1. Exploration des données

### 1. Combien de visites et de patients ?

| Jeu | Visites | Patients | Moyenne visites/patient | Médiane | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Train | 44 590 | 5 576 | 8,00 | 7 | 4 | 12 |
| Test | 11 013 | 1 395 | 7,89 | 7 | 4 | 12 |

Un patient typique possède donc **7 visites**, et le maximum est **12** dans
les deux jeux. La répétition des patients confirme le caractère longitudinal
des données et justifie une validation groupée.

### 2. Quelles colonnes sont les plus manquantes ?

Les trois colonnes les plus manquantes dans le train sont :

1. `time_since_intake_off` : **78,75 %** ;
2. `time_since_intake_on` : **46,37 %** ;
3. `off` : **42,42 %**.

`on` est absent sur **29,65 %** des visites. Il n'existe **aucune visite** où
`on` et `off` sont simultanément manquants. Chaque ligne possède donc au moins
une mesure clinique ON ou OFF.

### 3. Quelle est la distribution de `target` ?

`target` va de **0 à 109,5**, alors que l'échelle théorique MDS-UPDRS va de 0 à
132. Sa moyenne est `37,47`, sa médiane `37,3` et son écart-type `16,50`.
L'asymétrie mesurée est très faible (`skewness = 0,051`) : la distribution est
globalement symétrique, avec seulement une fine queue de valeurs très élevées.

### 4. Quelles variables sont associées à `target` ?

Les corrélations linéaires les plus fortes sont :

| Variable | Corrélation avec `target` |
|---|---:|
| `off` | 0,871 |
| `on` | 0,669 |
| `age` | 0,310 |
| `ledd` | 0,298 |
| `age_at_diagnosis` | 0,133 |

`off` est très informatif, mais ce n'est pas une copie de la cible : sur les
25 677 lignes où il est observé, son RMSE direct contre `target` vaut `10,20`,
son MAE `7,45` et il sous-estime la cible de `5,38` points en moyenne. `on` est
encore plus différent : RMSE direct `23,92` et sous-estimation moyenne de
`21,79` points. D'après la définition du challenge, ce sont des mesures
cliniques bruitées autorisées, pas les vraies cibles cachées.

### 5. `ledd` est-il manquant au hasard ?

Non, sa fréquence manquante dépend de la cohorte :

| Cohorte | Lignes | `ledd` manquant |
|---|---:|---:|
| A | 39 636 | 35,90 % |
| B | 4 954 | 42,59 % |

La différence de `6,69` points indique un effet de protocole ou de population.
Cliniquement, cela peut signifier que les cohortes ne suivent pas les mêmes
procédures de prescription ou de collecte. Les données seules ne permettent
pas de choisir entre ces explications : la valeur manquante doit donc être
conservée comme un signal potentiel, sans lui attribuer une causalité.

### 6. Quelles colonnes ne faut-il pas donner au modèle ?

- `target` : c'est la réponse à prédire ; l'inclure serait une fuite directe.
- `Index` : identifiant de ligne arbitraire sans signification clinique.
- `patient_id` : ne doit pas être une variable prédictive, car il permettrait de
  mémoriser les patients du train. Il reste indispensable comme groupe pour la
  validation.

`on`, `off` et `rater_id` peuvent être utilisés : ils décrivent le processus de
mesure. Il faut cependant contrôler leurs biais et toujours valider sur des
patients entièrement nouveaux.

## 2. Dummy mean

### 1. RMSE du Dummy et écart-type de la cible

Le Dummy en holdout obtient une RMSE de **16,4779**, presque identique à
l'écart-type de `target`, **16,4997**. C'est attendu : prédire constamment la
moyenne produit une erreur quadratique égale à la variance de la cible, à la
petite différence près entre la distribution du train et celle du holdout.

### 2. Forme du graphique d'erreur

Toutes les prédictions du Dummy sont proches de `37,47`. Les points forment
donc une ligne correspondant à une prédiction constante, tandis que les vraies
valeurs s'étalent de 0 à plus de 100. Un modèle parfait placerait tous les
points sur la diagonale `prédit = réel`.

### 3. À partir de quel gain l'amélioration est-elle crédible ?

Une baisse isolée de quelques centièmes ne suffit pas. Pour ce projet, un seuil
pratique est **au moins 0,5 point de RMSE**, avec un gain visible sur la plupart
des cinq folds patients. Le Ridge améliore le Dummy d'environ 5,9 points en CV
groupée : ce gain est très largement supérieur au bruit entre folds.

# Day 2

## 1. Ridge

### 1. Gain de Ridge contre le Dummy

Sur le même holdout aléatoire :

- Dummy : `16,4779` ;
- Ridge : `10,4877` ;
- baisse relative : **36,35 %**.

C'est un gain important : les variables cliniques contiennent beaucoup plus
d'information que la moyenne globale.

### 2. Coefficients principaux

Sans standardisation, le coefficient brut le plus grand en valeur absolue est
celui de `time_since_intake_on` (`+1,693`). Les coefficients de `on` (`+0,479`)
et `off` (`+0,591`) sont positifs, comme attendu : une mesure clinique plus
élevée doit généralement conduire à une cible plus élevée.

### 3. Pourquoi ne pas comparer directement les coefficients bruts ?

Les colonnes n'ont pas les mêmes unités ni les mêmes amplitudes. Un coefficient
de `0,001` appliqué à une dose pouvant atteindre plusieurs centaines peut avoir
plus d'effet qu'un coefficient de `1` appliqué à une variable variant peu. Il
faudrait standardiser les variables, ou comparer l'effet d'une variation d'un
écart-type, avant de parler d'importance.

## 2. Réglage de `alpha`

### 1. Variation du RMSE

| `alpha` | RMSE holdout |
|---:|---:|
| 0,01 | 10,4876591 |
| 0,1 | 10,4876591 |
| 1 | 10,4876594 |
| 10 | 10,4876619 |
| 100 | 10,4876888 |
| 1 000 | 10,4880928 |

Entre `0,1` et `10`, le mouvement est inférieur à `0,000003` : il est nul en
pratique. Avec huit variables et plus de 35 000 lignes d'apprentissage, les
coefficients sont déjà très stables et la pénalisation apporte peu.

### 2. Variation des coefficients

Ils changent à peine. Par exemple, entre `alpha=0,01` et `alpha=1 000`, le
coefficient de `off` passe seulement de `0,59089` à `0,59151`, celui de `on` de
`0,47884` à `0,47730`, et la norme de tous les coefficients de `2,3598` à
`2,3103`. Le réglage d'`alpha` n'est donc pas un levier utile ici.

## 3. Validation groupée par patient

### 1. Comparaison avec le split aléatoire

| Modèle | Holdout aléatoire | CV patient-groupée | Écart |
|---|---:|---:|---:|
| Dummy | 16,4779 | 16,5020 | +0,0241 |
| Ridge | 10,4877 | 10,5992 | +0,1115 |

Le Ridge change davantage que le Dummy, car il peut profiter légèrement de la
présence d'autres visites du même patient dans un split par lignes. L'écart
reste modéré pour ce modèle linéaire, mais il pourrait devenir beaucoup plus
grand avec des variables ou des modèles capables de mémoriser les patients.

### 2. Quel score faut-il désormais croire ?

Il faut privilégier la **CV groupée**, car le test Kaggle contient des patients
absents du train. Le score Kaggle exact de cette soumission n'est pas enregistré
dans le dépôt, donc la proximité numérique ne peut pas être vérifiée ici. Le
protocole groupé reste néanmoins celui qui reproduit correctement la séparation
du challenge.

## 4. HistGradientBoosting

### 1. HGBR bat-il Ridge sur chaque fold ?

Oui. Les RMSE HGBR numériques sont `7,824 / 7,936 / 7,576 / 7,617 / 7,824`,
contre `10,663 / 11,018 / 10,446 / 10,320 / 10,535` pour Ridge. Le gain n'est
donc pas causé par un seul fold favorable.

### 2. Temps de calcul

Sur ce recalcul local, un fold HGBR prend environ `0,27 s` pour ajustement et
prédiction, contre `0,024 s` pour Ridge, soit approximativement **11 fois plus**.
HGBR réduit cependant la RMSE de `10,60` à `7,76` : ce surcoût reste très faible
en valeur absolue et le gain justifie largement son utilisation.

### 3. Importance des variables

La permutation des variables sur le premier fold augmente surtout la RMSE pour
`off` (`+7,64`), `on` (`+5,37`), `age` (`+2,25`) et
`age_at_diagnosis` (`+1,79`). Les autres variables contribuent beaucoup moins.
Ce classement diffère des coefficients bruts du Ridge : l'arbre capture des
seuils et interactions non linéaires, tandis que les coefficients Ridge sont
fortement dépendants de l'unité des colonnes.

## 5. TableVectorizer + HGBR

### 1. Apport de `gene`, `cohort` et `rater_id`

Le HGBR numérique obtient `7,7568`, contre `7,5296` avec TableVectorizer, soit
un gain de **0,2272 RMSE**. Le modèle enrichi gagne sur chacun des cinq folds.
Le gain est modeste, mais supérieur à l'écart-type inter-fold du modèle enrichi
(`0,121`) et surtout régulier : les catégories apportent donc un vrai signal.

### 2. Encodeurs choisis

- `cohort` : `OneHotEncoder` (une colonne utile car la variable est binaire) ;
- `gene` : `OneHotEncoder` (quatre catégories plus une catégorie manquante) ;
- `rater_id` : `StringEncoder` produisant 30 composantes ;
- variables numériques : passage direct, NaN conservés pour HGBR.

Ce choix correspond aux cardinalités : `cohort` et `gene` sont faibles, alors
que les 50 raters constituent une catégorie plus large.

## 6. DataOps

### 1. Le score doit-il être identique à l'étape 5 ?

Il ne sera identique que si les colonnes, les folds et l'estimateur sont
strictement les mêmes. Or le code pédagogique de l'étape 6 fait seulement
`drop("target")` : il conserve donc `Index` et `patient_id`, contrairement à
l'étape 5. Une différence de score serait ici causée par ces colonnes, pas par
DataOps lui-même. Pour une comparaison correcte, il faut également retirer
`Index` et `patient_id` avant `mark_as_X`.

### 2. Que contient réellement `X_op` ?

Tel qu'écrit dans le guide, `X_op` contient toutes les variables de `visits`
sauf `target`, donc aussi `Index` et `patient_id`. Si ces identifiants figurent
dans l'importance des variables, cela signifie que le modèle exploite une
numérotation ou mémorise les patients. Ce serait un signal de fuite ou, au
minimum, d'apprentissage non généralisable. `patient_id` doit servir uniquement
à construire les groupes.

## 7. Bilan

### 1. Quel changement a le plus aidé ?

| Changement | Évolution approximative de RMSE |
|---|---:|
| Dummy → Ridge | 16,50 → 10,60 (`−5,90`) |
| Ridge → HGBR numérique | 10,60 → 7,76 (`−2,84`) |
| Ajout des catégories | 7,76 → 7,53 (`−0,23`) |
| Réglage d'`alpha` Ridge | pratiquement aucun gain |

Le premier modèle exploitant réellement les mesures cliniques donne le plus
gros gain. Ensuite, le passage au boosting et à ses non-linéarités est décisif.
La CV groupée n'est pas censée améliorer le modèle : elle rend simplement
l'estimation honnête.

### 2. Qu'ont en commun les plus grosses erreurs ?

Les erreurs se concentrent sur les situations difficiles suivantes :

- valeurs sévères : pour `target > 70`, la RMSE atteint **14,27** ;
- visites `on_only`, donc sans mesure `off` : RMSE **8,43**, contre **5,90**
  pour `off_only` ;
- maladie avancée : RMSE **8,45** dans le dernier quartile de durée depuis le
  diagnostic, contre **6,49** dans le premier ;
- cohorte A : RMSE **7,65**, contre **6,62** pour B ;
- certains raters, notamment R03 (`8,40`), R15 (`8,28`) et R01 (`8,12`).

Le cas récurrent `ETNU0107` possède des cibles proches de 100–109, mais des
prédictions autour de 68–70 : le modèle régresse trop fortement vers la moyenne
pour les patients extrêmement sévères. Les quartiles de temps depuis la prise
ont en revanche des RMSE proches ; la longueur du délai seule n'explique pas les
plus grosses erreurs.

### 3. Quel problème reste à traiter ?

Le Day 2 traite partiellement la **missingness**, mais ne modélise pas vraiment
la **trajectoire temporelle d'un patient** ni la forme pharmacocinétique de la
réponse au traitement. La prochaine étape logique est de créer :

- `disease_duration = age - age_at_diagnosis` ;
- des indicateurs `on_only`, `off_only` et `both` ;
- des transformations non linéaires des délais depuis la prise ;
- des caractéristiques longitudinales par patient calculées sans cible et de
  façon sûre à l'intérieur de chaque fold.

Ces variables visent directement les visites ON-only, les stades avancés et les
effets du médicament identifiés par l'analyse d'erreurs.

## Sources locales utilisées

- `data/eda.md` et les quatre CSV officiels du dossier `data/` ;
- `experiments/01_dummy.py`, `02_ridge.py` et `03_day2.py` ;
- `results/day3_diagnostics/` pour les prédictions OOF et erreurs détaillées ;
- `docs/DAY1.md`, `docs/DAY2.md` et `docs/CONTEXT.md` pour les questions et le
  contexte clinique.
