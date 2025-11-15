# Federated-Learning-P1

## Article

[Mean Aggregator is More Robust than Robust Aggregators under Label Poisoning Attacks on Distributed Heterogeneous Data](https://arxiv.org/abs/2404.13647)

## Overleaf

https://www.overleaf.com/7776336468zwtkjqjgxkmb#b11253
 
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

## Structure du Projet

Le projet est organisé en plusieurs modules Python qui séparent les différentes logiques de la simulation.

- **`main_MNIST.py`**: Script principal pour lancer des batteries d'expériences. Il itère sur différentes partitions de données, types d'attaques et modèles, puis sauvegarde les résultats (métriques et graphiques).
- **`main_inter.py`**: Script spécialisé pour obtenir des résultats statistiquement robustes (avec intervalles de confiance) pour une configuration d'attaque/modèle donnée, en répétant la simulation plusieurs fois.
- **`simu.py`**: Contient le moteur de la simulation (`run_simulation`). Cette fonction orchestre l'entraînement fédéré sur plusieurs tours de communication, de la distribution des données aux clients à l'agrégation des mises à jour.
- **`worker.py`**: Définit la classe `Worker` qui simule un client. Chaque client possède ses propres données locales et est capable de calculer une mise à jour du modèle (gradient).
- **`aggregators.py`**: Implémente les différentes fonctions d'agrégation (Mean, TriMean, FABA, etc.) qui combinent les mises à jour des clients au niveau du serveur.
- **`data_partition.py`**: Fournit des fonctions pour distribuer les données d'entraînement entre les clients selon différentes stratégies (IID, Dirichlet, non-IID pathologique, etc.) pour simuler l'hétérogénéité des données.
- **`models.py`**: Définit les architectures des modèles neuronaux utilisés (Softmax, MLP, CNN).
- **`label_poisoning.py`**: Contient l'implémentation des différentes stratégies d'attaque, qu'elles soient au niveau des données (label-flipping) ou des gradients (sign-flipping, model replacement).
- **`plot.py`**: Regroupe des fonctions utilitaires pour générer les graphiques à partir des résultats des simulations.

---

## Comment lancer les simulations ?

### 1. Lancement d'une batterie d'expériences (`main_MNIST.py`)

Ce script est idéal pour comparer rapidement l'impact de différentes configurations.

**Fonctionnement :**
1.  Il définit des listes de configurations à tester : `partition_list`, `attack_list`, `model_list`.
2.  Il boucle sur chaque combinaison (modèle, attaque, partition).
3.  Pour chaque combinaison, il appelle `run_simulation` et stocke les métriques (accuracy, loss, variance, xi, A).
4.  À la fin de chaque type d'attaque, il génère deux types de graphiques :
    -   `{ATTACK}_aggregator_comparison_partitions_mnist.png`: Compare les performances des agrégateurs pour chaque partition de données.
    -   `{ATTACK}_xi_A_comparison_partitions.png`: Compare les métriques d'hétérogénéité (ξ) et de perturbation (A) pour chaque partition.
5.  Les résultats bruts sont sauvegardés dans un fichier CSV consolidé.

**Pour l'exécuter :**

```bash
python main_MNIST.py
```

Les résultats sont sauvegardés dans les dossiers `plots/` et `data_results/`, organisés par date et par modèle.

### 2. Lancement d'une simulation avec intervalles de confiance (`main_inter.py`)

Ce script est conçu pour produire des graphiques plus fiables en répétant `N` fois la même simulation pour lisser les effets aléatoires (initialisation du modèle, échantillonnage des données).

**Fonctionnement :**
1.  Fixez une configuration unique (un seul `ATTACK` et `MODEL`).
2.  Le script exécute la simulation `N` fois (par exemple, `N=10`).
3.  Il accumule les résultats de chaque exécution.
4.  Il calcule la **moyenne** et l'**intervalle de confiance à 95%** pour chaque métrique à chaque tour de communication.
5.  Il génère des graphiques montrant la performance moyenne des agrégateurs, avec une bande d'incertitude.

**Pour l'exécuter :**

```bash
python main_inter.py
```

Les graphiques et le CSV contenant les moyennes et les intervalles de confiance sont également sauvegardés dans `plots/` et `data_results/`.

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

Ce projet vise à tester, vérifier, améliorer et étendre les expériences de l'article de référence.

## Tâches


| Tâche                                                        | Alexander | Noé | Cédric | Andrea | DONE |
|--------------------------------------------------------------|:----------:|:---:|:-------:|:--------------:|:-------:|
| Implémenter CNN                                              | ✅ |  |  |  | YES |
| Organiser le code                                            | ✅ |  |  |  | YES |
| Voir démos et lire en détail l'article et les arguments      |  | ✅ | ✅ |  | En cours|
| Améliorer le plot                                            | ✅ |  |  |  | YES |
| Ajouter la variance                                          | ✅ |  |  |  | YES |
| Ajouter intervalles de confiance des valeurs                 | ✅ |  |  |  | YES |
| Optimiser GPU et vectoriser le code                          | ✅ |  |  |  | En cours |
| Apporter les grandes idées des théorèmes avec dessins        |  | ✅ | ✅ |  | En cours |
| Explorer d'autres aggregators                                |  | ✅ |  |  |  |
| Améliorer les stratégies d'attaques (byzantin)               | ✅ | ✅ |  |  | YES |
| Tester théoriquement quand on change de modèles              |  |  |  | ✅ | En cours pour moins d'hétérogénéité ($A$ petit) |
| Tester théoriquement quand on change de stratégies           |  |  |  | ✅ |  |
| Coder avec d'autres types de dataset                         | ✅ |  |  |  |  |
| Coder d'autres distributions de dataset                      | ✅ |  |  |  |  |
| Apporter des idées de dataset                                | ✅ | ✅ | ✅ | ✅ |  |
| Tester le cas $A\delta < \rho \xi$ (pas de supériorité du mean a priori)                         |  |  |  |  |  |

---

## Licence

Ecole Polytechnique
