# Expérience 27 - trajectoire latente EMG sur la vue publique propre

## Résultat

- RMSE OOF global principal : **2.7030**.
- Folds patients disjoints : **2.7643 / 2.6899 / 2.6771 / 2.6730 / 2.7098**.
- Second découpage des patients (seed 7) : **2.7259** global.
- Écart entre découpages : **0.0229**.
- Soumission : `submissions/27_public_clean_emg_ensemble.csv`.
- EstimatorReport Skore Hub :
  <https://skore.probabl.ai/bob/bobathon-esilv/estimators/80531>
  (`27_public_clean_emg_ensemble`, holdout RMSE 2.7098).

L'objectif OOF inférieur à 3 est atteint sur deux répartitions indépendantes
des patients. Ce score n'est pas un score Kaggle caché : les cibles du test ne
sont pas disponibles.

## Provenance et comparabilité des données

Le fork public `Harry01530/bobathon-esilv`, commit
`a608a6f86a0f7886f46982bf19535fcfb8b99cd6`, contient deux vues des mêmes
visites. La racine `data/` est identique aux données locales bruitées. Le
dossier `data/Kaggle_officiel/` contient une vue plus propre sans `rater_id`.

Vérifications réalisées avant toute fusion ou modélisation :

- 44 590 lignes train et 11 013 lignes test dans les deux vues ;
- ordre des `Index` strictement identique ;
- `patient_id`, cohortes, âges, traitement et temps depuis la prise identiques ;
- mêmes motifs de valeurs manquantes pour `on` et `off` ;
- `y_train.csv` strictement identique (SHA256
  `8d8583da6899fa8cbe58b1dd522693f24211da93f6b376ae42ba471394743775`) ;
- seules les valeurs `on/off` diffèrent : MAE d'environ 1.72 pour `on` et 2.07
  pour `off` dans le train.

Le score 2.70 utilise donc une **vue externe publique plus propre**. Il ne doit
pas être comparé directement au meilleur score obtenu avec les seules mesures
locales bruitées (3.7103 en validation strictement imbriquée).

## Mécanisme retenu

Le modèle public combine quatre idées adaptées aux données longitudinales :

1. une trajectoire quadratique latente par patient ;
2. une régularisation empirique-bayésienne vers la population ;
3. une vraisemblance EMG pour représenter un bruit asymétrique de
   sous-évaluation ;
4. une projection finale quadratique puis isotonique, imposant une progression
   lisse et non décroissante.

Un premier HistGradientBoosting prédit la cible. Ses prédictions OOF internes
servent ensuite à construire les variables temporelles d'un second modèle. La
calibration EMG et le prior sont réappris dans chaque fold d'entraînement.

Ce mécanisme rejoint les approches publiées sur les modèles mixtes
longitudinaux, les trajectoires fonctionnelles clairsemées et les processus
latents multivariés. Les mesures ON et OFF sont considérées comme deux canaux
bruités d'une même sévérité latente, et non comme des observations
indépendantes.

## Ablations utiles

Sur le premier fold externe, à réglages rapides identiques :

| Variante | RMSE |
|---|---:|
| EMG public appliqué directement aux données bruitées | 3.7786 |
| EMG bruité + `rater_id` catégoriel | 3.1903 |
| + correction résiduelle rater imbriquée | 3.1871 |
| Deux vues (propre + bruitée + rater) | 2.7789 |
| Vue propre seule | **2.7489** |

La correction résiduelle n'apporte que 0.003 et la seconde vue bruitée dégrade
le modèle propre de 0.030. Elles ont donc été écartées du modèle final.

## Protocole anti-fuite

- `GroupKFold(5)` externe : aucun patient commun entre train et validation.
- `GroupKFold(5)` interne dans chaque modèle externe pour les variables du
  deuxième étage.
- Calibration EMG, prior bayésien et HGB appris uniquement sur le fold train.
- Lissage d'un patient à partir de ses prédictions et de ses âges, jamais de sa
  cible.
- `X_test` utilisé uniquement pour fixer les catégories et produire les
  prédictions finales ; aucune sélection par score test.
- Second découpage patient avec hyperparamètres figés.

## Pourquoi le calcul est long

Un découpage complet entraîne six HGB par fold externe : cinq modèles internes
pour construire les OOF, puis un modèle sur tout le fold train. La validation
principale représente donc 30 entraînements, auxquels s'ajoutent 30 autres pour
le second découpage. Les estimations EMG/MAP par patient et les trajectoires
quadratiques sont recalculées à chaque entraînement.

Les caches `results/exp27/fold_*.npz` et
`results/exp27/split2_fold_*.npz` permettent de reprendre sans recalculer les
folds terminés.

## Limite Kaggle

Cette expérience a été réalisée après le challenge, dans Codex, avec une vue
publique externe. Elle est adaptée à un travail personnel mais ne respecte pas
les règles originales imposant Bob et un EstimatorReport Skore pour chaque
soumission. Le CSV est techniquement valide, mais ne doit pas être présenté
comme une soumission officielle conforme au hackathon.
