# Federated-Learning-P1

## Article

[Mean Aggregator is More Robust than Robust Aggregators under Label Poisoning Attacks on Distributed Heterogeneous Data](https://arxiv.org/abs/2404.13647)

### Résumé de l’article

Cet article remet en question une idée reçue dans l’apprentissage fédéré : les **agrégateurs dits "robustes"** (comme Median, Trimmed Mean ou Krum) ne sont pas nécessairement plus performants que la **moyenne simple (Mean Aggregator)** face à certaines attaques, notamment les **attaques de type "Label Poisoning"** sur des données **non-i.i.d. (hétérogènes)**.

Les auteurs montrent que :
- Dans des contextes **distribués et hétérogènes**, la robustesse des agrégateurs robustes peut se dégrader.
- Le **Mean Aggregator** (FedAvg classique) peut mieux tolérer les perturbations induites par des clients malicieux lorsque la variabilité inter-clients est déjà forte.
- Les attaques de **Label Poisoning** (où certains clients changent les étiquettes des données d’entraînement) ne modifient pas nécessairement les gradients d’une façon que les méthodes robustes détectent efficacement.

En résumé :
> La moyenne simple, bien que naïve, reste étonnamment stable et efficace dans des scénarios de données non homogènes avec attaques label-poisoning.

### Contributions principales

1. **Analyse théorique** : les auteurs montrent pourquoi la moyenne simple reste compétitive dans des environnements non i.i.d.
2. **Études expérimentales** : sur plusieurs datasets (comme MNIST, CIFAR-10, FEMNIST), le Mean Aggregator surpasse des agrégateurs robustes sous attaques label-poisoning.
3. **Discussion pratique** : les méthodes robustes supposent souvent une homogénéité des données entre clients, hypothèse rarement vraie dans la pratique.

---

# Implémentation des agrégateurs

Le fichier contient plusieurs **méthodes d’agrégation** (`aggregators`) qui combinent les mises à jour (ou gradients) locales des clients lors de l’entraînement fédéré.  
Chaque méthode tente de limiter l’influence des **clients malicieux** ou **bruyants**.

## 1. `Mean` — Moyenne simple
Principe : calcule la moyenne de toutes les mises à jour locales.

Avantage : simple, rapide et souvent performant dans des environnements hétérogènes.

Limite : sensible aux valeurs aberrantes (outliers) et aux attaques byzantines.

## 2. `TriMean` — Moyenne tronquée
Principe : pour chaque coordonnée, on supprime les valeurs extrêmes (les plus petites et les plus grandes) selon un ratio trim_ratio, puis on fait la moyenne sur le reste.

Avantage : robuste face à quelques mises à jour extrêmes.

Limite : perd de l’information utile si beaucoup de clients honnêtes ont des valeurs éloignées (cas de données non-i.i.d.).

## 3. `CoordMedian` — Médiane coordonnée
Principe : pour chaque dimension (poids du modèle), on prend la médiane au lieu de la moyenne.

Avantage : résiste fortement aux outliers.

Limite : peut ignorer des tendances réelles si les gradients sont très dispersés entre clients honnêtes.

## 4. `CC` — Centered Clipping
Principe : on calcule la moyenne, puis on “clipse” (réduit) les mises à jour trop éloignées de cette moyenne.

Avantage : limite l’impact des clients avec de grands gradients anormaux.

Limite : nécessite de choisir un seuil clip_threshold adapté.


## 5. `FABA` — Fooling-Attack Byzantine Averaging (simple)
Principe : retire itérativement les clients dont la mise à jour est la plus éloignée de la moyenne courante, selon une fraction remove_frac.

Avantage : élimine progressivement les clients suspects.

Limite : coûteux si beaucoup de clients et sensible au choix de la fraction à retirer.

## 6. `LFighter` — Large-cluster Fighter
Principe : utilise le clustering (KMeans) pour regrouper les mises à jour, puis garde le plus grand cluster (supposé contenir les clients honnêtes).

Avantage : efficace si les mises à jour malicieuses forment un petit groupe isolé.

Limite : dépend du nombre de clusters et de la distribution des données ; sensible au bruit.

---

## Missions

Tester, vérifier, améliorer et étendre

## Lancement du programme

Aller sur 
```bash
main.py
```

## Tâches


| Tâche                                                        | Alexander | Noé | Cédric | Andrea | DONE |
|--------------------------------------------------------------|:----------:|:---:|:-------:|:--------------:|:-------:|
| Implémenter CNN                                              | ✅ |  |  |  | YES |
| Organiser le code                                            | ✅ |  |  |  | YES |
| Voir démos et lire en détail l'article et les arguments      |  | ✅ | ✅ |  |  |
| Améliorer le plot                                            | ✅ |  |  |  | YES |
| Ajouter la variance                                          | ✅ |  |  |  | YES |
| Ajouter intervalles de confiance des valeurs                 | ✅ |  |  |  | YES |
| Optimiser GPU et vectoriser le code                          | ✅ |  |  |  | En cours |
| Apporter les grandes idées des théorèmes avec dessins        |  | ✅ | ✅ |  |  |
| Explorer d'autres aggregators                                |  | ✅ |  |  |  |
| Améliorer les stratégies d'attaques (byzantin)               | ✅ | ✅ |  |  | YES |
| Tester théoriquement quand on change de modèles              |  |  |  | ✅ |  |
| Tester théoriquement quand on change de stratégies           |  |  |  | ✅ |  |
| Coder avec d'autres types de dataset                         | ✅ |  |  |  |  |
| Coder d'autres distributions de dataset                      | ✅ |  |  |  |  |
| Apporter des idées de dataset                                | ✅ | ✅ | ✅ | ✅ |  |

---

## Licence

Ecole Polytechnique