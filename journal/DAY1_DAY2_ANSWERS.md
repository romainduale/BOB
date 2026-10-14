# R├®ponses au workbook ÔÇö Day 1 et Day 2

Ce document r├®pond aux questions de `docs/DAY1.md` et `docs/DAY2.md` ├á
partir des donn├®es officielles locales du projet. Les scores ont ├®t├® recalcul├®s
avec les scripts et d├®pendances actuellement pr├®sents dans le d├®p├┤t.

## Rep├¿res de validation

- **Holdout al├®atoire 80/20** : utile pour les premiers contr├┤les, mais des
  visites d'un m├¬me patient peuvent se retrouver des deux c├┤t├®s.
- **GroupKFold ├á 5 folds par `patient_id`** : estimation ├á privil├®gier, car un
  patient est enti├¿rement dans le train ou dans la validation.
- Les valeurs recalcul├®es pour le Day 2 sont : Dummy `16,50`, Ridge `10,60`,
  HGBR num├®rique `7,76` et TableVectorizer + HGBR `7,53`.
- Le fichier OOF historique de l'analyse d'erreurs donne `7,54`. L'├®cart de
  `0,01` avec le recalcul (`7,53`) est n├®gligeable et peut venir des versions de
  biblioth├¿ques ou du run sauvegard├®.

# Day 1

## 1. Exploration des donn├®es

### 1. Combien de visites et de patients ?

| Jeu | Visites | Patients | Moyenne visites/patient | M├®diane | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| Train | 44 590 | 5 576 | 8,00 | 7 | 4 | 12 |
| Test | 11 013 | 1 395 | 7,89 | 7 | 4 | 12 |

Un patient typique poss├¿de donc **7 visites**, et le maximum est **12** dans
les deux jeux. La r├®p├®tition des patients confirme le caract├¿re longitudinal
des donn├®es et justifie une validation group├®e.

### 2. Quelles colonnes sont les plus manquantes ?

Les trois colonnes les plus manquantes dans le train sont :

1. `time_since_intake_off` : **78,75 %** ;
2. `time_since_intake_on` : **46,37 %** ;
3. `off` : **42,42 %**.

`on` est absent sur **29,65 %** des visites. Il n'existe **aucune visite** o├╣
`on` et `off` sont simultan├®ment manquants. Chaque ligne poss├¿de donc au moins
une mesure clinique ON ou OFF.

### 3. Quelle est la distribution de `target` ?

`target` va de **0 ├á 109,5**, alors que l'├®chelle th├®orique MDS-UPDRS va de 0 ├á
132. Sa moyenne est `37,47`, sa m├®diane `37,3` et son ├®cart-type `16,50`.
L'asym├®trie mesur├®e est tr├¿s faible (`skewness = 0,051`) : la distribution est
globalement sym├®trique, avec seulement une fine queue de valeurs tr├¿s ├®lev├®es.

### 4. Quelles variables sont associ├®es ├á `target` ?

Les corr├®lations lin├®aires les plus fortes sont :

| Variable | Corr├®lation avec `target` |
|---|---:|
| `off` | 0,871 |
| `on` | 0,669 |
| `age` | 0,310 |
| `ledd` | 0,298 |
| `age_at_diagnosis` | 0,133 |

`off` est tr├¿s informatif, mais ce n'est pas une copie de la cible : sur les
25 677 lignes o├╣ il est observ├®, son RMSE direct contre `target` vaut `10,20`,
son MAE `7,45` et il sous-estime la cible de `5,38` points en moyenne. `on` est
encore plus diff├®rent : RMSE direct `23,92` et sous-estimation moyenne de
`21,79` points. D'apr├¿s la d├®finition du challenge, ce sont des mesures
cliniques bruit├®es autoris├®es, pas les vraies cibles cach├®es.

### 5. `ledd` est-il manquant au hasard ?

Non, sa fr├®quence manquante d├®pend de la cohorte :

| Cohorte | Lignes | `ledd` manquant |
|---|---:|---:|
| A | 39 636 | 35,90 % |
| B | 4 954 | 42,59 % |

La diff├®rence de `6,69` points indique un effet de protocole ou de population.
Cliniquement, cela peut signifier que les cohortes ne suivent pas les m├¬mes
proc├®dures de prescription ou de collecte. Les donn├®es seules ne permettent
pas de choisir entre ces explications : la valeur manquante doit donc ├¬tre
conserv├®e comme un signal potentiel, sans lui attribuer une causalit├®.

### 6. Quelles colonnes ne faut-il pas donner au mod├¿le ?

- `target` : c'est la r├®ponse ├á pr├®dire ; l'inclure serait une fuite directe.
- `Index` : identifiant de ligne arbitraire sans signification clinique.
- `patient_id` : ne doit pas ├¬tre une variable pr├®dictive, car il permettrait de
  m├®moriser les patients du train. Il reste indispensable comme groupe pour la
  validation.

`on`, `off` et `rater_id` peuvent ├¬tre utilis├®s : ils d├®crivent le processus de
mesure. Il faut cependant contr├┤ler leurs biais et toujours valider sur des
patients enti├¿rement nouveaux.

## 2. Dummy mean

### 1. RMSE du Dummy et ├®cart-type de la cible

Le Dummy en holdout obtient une RMSE de **16,4779**, presque identique ├á
l'├®cart-type de `target`, **16,4997**. C'est attendu : pr├®dire constamment la
moyenne produit une erreur quadratique ├®gale ├á la variance de la cible, ├á la
petite diff├®rence pr├¿s entre la distribution du train et celle du holdout.

### 2. Forme du graphique d'erreur

Toutes les pr├®dictions du Dummy sont proches de `37,47`. Les points forment
donc une ligne correspondant ├á une pr├®diction constante, tandis que les vraies
valeurs s'├®talent de 0 ├á plus de 100. Un mod├¿le parfait placerait tous les
points sur la diagonale `pr├®dit = r├®el`.

### 3. ├Ç partir de quel gain l'am├®lioration est-elle cr├®dible ?

Une baisse isol├®e de quelques centi├¿mes ne suffit pas. Pour ce projet, un seuil
pratique est **au moins 0,5 point de RMSE**, avec un gain visible sur la plupart
des cinq folds patients. Le Ridge am├®liore le Dummy d'environ 5,9 points en CV
group├®e : ce gain est tr├¿s largement sup├®rieur au bruit entre folds.

# Day 2

## 1. Ridge

### 1. Gain de Ridge contre le Dummy

Sur le m├¬me holdout al├®atoire :

- Dummy : `16,4779` ;
- Ridge : `10,4877` ;
- baisse relative : **36,35 %**.

C'est un gain important : les variables cliniques contiennent beaucoup plus
d'information que la moyenne globale.

### 2. Coefficients principaux

Sans standardisation, le coefficient brut le plus grand en valeur absolue est
celui de `time_since_intake_on` (`+1,693`). Les coefficients de `on` (`+0,479`)
et `off` (`+0,591`) sont positifs, comme attendu : une mesure clinique plus
├®lev├®e doit g├®n├®ralement conduire ├á une cible plus ├®lev├®e.

### 3. Pourquoi ne pas comparer directement les coefficients bruts ?

Les colonnes n'ont pas les m├¬mes unit├®s ni les m├¬mes amplitudes. Un coefficient
de `0,001` appliqu├® ├á une dose pouvant atteindre plusieurs centaines peut avoir
plus d'effet qu'un coefficient de `1` appliqu├® ├á une variable variant peu. Il
faudrait standardiser les variables, ou comparer l'effet d'une variation d'un
├®cart-type, avant de parler d'importance.

## 2. R├®glage de `alpha`

### 1. Variation du RMSE

| `alpha` | RMSE holdout |
|---:|---:|
| 0,01 | 10,4876591 |
| 0,1 | 10,4876591 |
| 1 | 10,4876594 |
| 10 | 10,4876619 |
| 100 | 10,4876888 |
| 1 000 | 10,4880928 |

Entre `0,1` et `10`, le mouvement est inf├®rieur ├á `0,000003` : il est nul en
pratique. Avec huit variables et plus de 35 000 lignes d'apprentissage, les
coefficients sont d├®j├á tr├¿s stables et la p├®nalisation apporte peu.

### 2. Variation des coefficients

Ils changent ├á peine. Par exemple, entre `alpha=0,01` et `alpha=1 000`, le
coefficient de `off` passe seulement de `0,59089` ├á `0,59151`, celui de `on` de
`0,47884` ├á `0,47730`, et la norme de tous les coefficients de `2,3598` ├á
`2,3103`. Le r├®glage d'`alpha` n'est donc pas un levier utile ici.

## 3. Validation group├®e par patient

### 1. Comparaison avec le split al├®atoire

| Mod├¿le | Holdout al├®atoire | CV patient-group├®e | ├ëcart |
|---|---:|---:|---:|
| Dummy | 16,4779 | 16,5020 | +0,0241 |
| Ridge | 10,4877 | 10,5992 | +0,1115 |

Le Ridge change davantage que le Dummy, car il peut profiter l├®g├¿rement de la
pr├®sence d'autres visites du m├¬me patient dans un split par lignes. L'├®cart
reste mod├®r├® pour ce mod├¿le lin├®aire, mais il pourrait devenir beaucoup plus
grand avec des variables ou des mod├¿les capables de m├®moriser les patients.

### 2. Quel score faut-il d├®sormais croire ?

Il faut privil├®gier la **CV group├®e**, car le test Kaggle contient des patients
absents du train. Le score Kaggle exact de cette soumission n'est pas enregistr├®
dans le d├®p├┤t, donc la proximit├® num├®rique ne peut pas ├¬tre v├®rifi├®e ici. Le
protocole group├® reste n├®anmoins celui qui reproduit correctement la s├®paration
du challenge.

## 4. HistGradientBoosting

### 1. HGBR bat-il Ridge sur chaque fold ?

Oui. Les RMSE HGBR num├®riques sont `7,824 / 7,936 / 7,576 / 7,617 / 7,824`,
contre `10,663 / 11,018 / 10,446 / 10,320 / 10,535` pour Ridge. Le gain n'est
donc pas caus├® par un seul fold favorable.

### 2. Temps de calcul

Sur ce recalcul local, un fold HGBR prend environ `0,27 s` pour ajustement et
pr├®diction, contre `0,024 s` pour Ridge, soit approximativement **11 fois plus**.
HGBR r├®duit cependant la RMSE de `10,60` ├á `7,76` : ce surco├╗t reste tr├¿s faible
en valeur absolue et le gain justifie largement son utilisation.

### 3. Importance des variables

La permutation des variables sur le premier fold augmente surtout la RMSE pour
`off` (`+7,64`), `on` (`+5,37`), `age` (`+2,25`) et
`age_at_diagnosis` (`+1,79`). Les autres variables contribuent beaucoup moins.
Ce classement diff├¿re des coefficients bruts du Ridge : l'arbre capture des
seuils et interactions non lin├®aires, tandis que les coefficients Ridge sont
fortement d├®pendants de l'unit├® des colonnes.

## 5. TableVectorizer + HGBR

### 1. Apport de `gene`, `cohort` et `rater_id`

Le HGBR num├®rique obtient `7,7568`, contre `7,5296` avec TableVectorizer, soit
un gain de **0,2272 RMSE**. Le mod├¿le enrichi gagne sur chacun des cinq folds.
Le gain est modeste, mais sup├®rieur ├á l'├®cart-type inter-fold du mod├¿le enrichi
(`0,121`) et surtout r├®gulier : les cat├®gories apportent donc un vrai signal.

### 2. Encodeurs choisis

- `cohort` : `OneHotEncoder` (une colonne utile car la variable est binaire) ;
- `gene` : `OneHotEncoder` (quatre cat├®gories plus une cat├®gorie manquante) ;
- `rater_id` : `StringEncoder` produisant 30 composantes ;
- variables num├®riques : passage direct, NaN conserv├®s pour HGBR.

Ce choix correspond aux cardinalit├®s : `cohort` et `gene` sont faibles, alors
que les 50 raters constituent une cat├®gorie plus large.

## 6. DataOps

### 1. Le score doit-il ├¬tre identique ├á l'├®tape 5 ?

Il ne sera identique que si les colonnes, les folds et l'estimateur sont
strictement les m├¬mes. Or le code p├®dagogique de l'├®tape 6 fait seulement
`drop("target")` : il conserve donc `Index` et `patient_id`, contrairement ├á
l'├®tape 5. Une diff├®rence de score serait ici caus├®e par ces colonnes, pas par
DataOps lui-m├¬me. Pour une comparaison correcte, il faut ├®galement retirer
`Index` et `patient_id` avant `mark_as_X`.

### 2. Que contient r├®ellement `X_op` ?

Tel qu'├®crit dans le guide, `X_op` contient toutes les variables de `visits`
sauf `target`, donc aussi `Index` et `patient_id`. Si ces identifiants figurent
dans l'importance des variables, cela signifie que le mod├¿le exploite une
num├®rotation ou m├®morise les patients. Ce serait un signal de fuite ou, au
minimum, d'apprentissage non g├®n├®ralisable. `patient_id` doit servir uniquement
├á construire les groupes.

## 7. Bilan

### 1. Quel changement a le plus aid├® ?

| Changement | ├ëvolution approximative de RMSE |
|---|---:|
| Dummy ÔåÆ Ridge | 16,50 ÔåÆ 10,60 (`ÔêÆ5,90`) |
| Ridge ÔåÆ HGBR num├®rique | 10,60 ÔåÆ 7,76 (`ÔêÆ2,84`) |
| Ajout des cat├®gories | 7,76 ÔåÆ 7,53 (`ÔêÆ0,23`) |
| R├®glage d'`alpha` Ridge | pratiquement aucun gain |

Le premier mod├¿le exploitant r├®ellement les mesures cliniques donne le plus
gros gain. Ensuite, le passage au boosting et ├á ses non-lin├®arit├®s est d├®cisif.
La CV group├®e n'est pas cens├®e am├®liorer le mod├¿le : elle rend simplement
l'estimation honn├¬te.

### 2. Qu'ont en commun les plus grosses erreurs ?

Les erreurs se concentrent sur les situations difficiles suivantes :

- valeurs s├®v├¿res : pour `target > 70`, la RMSE atteint **14,27** ;
- visites `on_only`, donc sans mesure `off` : RMSE **8,43**, contre **5,90**
  pour `off_only` ;
- maladie avanc├®e : RMSE **8,45** dans le dernier quartile de dur├®e depuis le
  diagnostic, contre **6,49** dans le premier ;
- cohorte A : RMSE **7,65**, contre **6,62** pour B ;
- certains raters, notamment R03 (`8,40`), R15 (`8,28`) et R01 (`8,12`).

Le cas r├®current `ETNU0107` poss├¿de des cibles proches de 100ÔÇô109, mais des
pr├®dictions autour de 68ÔÇô70 : le mod├¿le r├®gresse trop fortement vers la moyenne
pour les patients extr├¬mement s├®v├¿res. Les quartiles de temps depuis la prise
ont en revanche des RMSE proches ; la longueur du d├®lai seule n'explique pas les
plus grosses erreurs.

### 3. Quel probl├¿me reste ├á traiter ?

Le Day 2 traite partiellement la **missingness**, mais ne mod├®lise pas vraiment
la **trajectoire temporelle d'un patient** ni la forme pharmacocin├®tique de la
r├®ponse au traitement. La prochaine ├®tape logique est de cr├®er :

- `disease_duration = age - age_at_diagnosis` ;
- des indicateurs `on_only`, `off_only` et `both` ;
- des transformations non lin├®aires des d├®lais depuis la prise ;
- des caract├®ristiques longitudinales par patient calcul├®es sans cible et de
  fa├ºon s├╗re ├á l'int├®rieur de chaque fold.

Ces variables visent directement les visites ON-only, les stades avanc├®s et les
effets du m├®dicament identifi├®s par l'analyse d'erreurs.

## Sources locales utilis├®es

- `data/eda.md` et les quatre CSV officiels du dossier `data/` ;
- `experiments/01_dummy.py`, `02_ridge.py` et `03_day2.py` ;
- `results/day3_diagnostics/` pour les pr├®dictions OOF et erreurs d├®taill├®es ;
- `docs/DAY1.md`, `docs/DAY2.md` et `docs/CONTEXT.md` pour les questions et le
  contexte clinique.
