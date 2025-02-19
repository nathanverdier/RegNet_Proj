# Le code

Le code implémente les composants suivants :

- **ConvLSTMCell**  
  Une cellule ConvLSTM qui traite des cartes de caractéristiques spatiales. La cellule concatène l'entrée courante avec l'état caché précédent, applique une convolution pour calculer les portes de l'LSTM, puis met à jour l'état de la cellule et l'état caché. Un contrôle d'erreur vérifie que les dimensions de l'entrée et de l'état caché correspondent.

- **RegNetBlock**  
  Un bloc régulé combinant un bloc de convolution classique avec un module ConvLSTM. Ce bloc comprend :  
  - Une première convolution avec gestion du stride, suivie d'une normalisation par lots et d'une activation ReLU.  
  - Un module ConvLSTM (si activé) qui traite la sortie de la première convolution, dont la sortie est ensuite concaténée avec les caractéristiques convolutionnelles et fusionnée via une convolution 1×1 suivie d'une normalisation et d'une activation.  
  - Une deuxième convolution avec normalisation par lots.  
  - Une connexion résiduelle qui peut inclure un ajustement (downsampling) lorsque le nombre de canaux change ou lorsque le stride est différent de 1.

- **Architecture RegNet**  
  Deux architectures sont proposées :  
  - **RegNetCIFAR** : Conçue pour des images de 32×32 (ex. CIFAR-10). Elle comporte une couche d'entrée, trois groupes de RegNetBlocks, une couche de pooling adaptatif et une couche entièrement connectée pour la classification.  
  - **RegNetImageNet** : Conçue pour des images de 224×224 (ex. ImageNet). Elle utilise une première convolution 7×7 suivie d'un max-pooling, puis quatre groupes de RegNetBlocks, un pooling adaptatif et une couche entièrement connectée.

- **Fonctions d'Entraînement et de Test**  
  Les fonctions `train_model` et `test_model` gèrent la boucle d'entraînement et l'évaluation du modèle, en affichant la perte et la précision pendant l'entraînement.

- **Interface en Ligne de Commande**  
  Le programme principal accepte des arguments pour sélectionner le jeu de données (CIFAR-10 ou ImageNet), spécifier le chemin des données, le nombre d'époques, la taille du batch et le taux d'apprentissage.

## Prérequis

- Python 3.x  
- PyTorch  
- Torchvision  
- argparse

Installez les packages requis avec pip :

```bash
pip install -r requirements.txt
```

## Utilisation

Exécutez le programme principal avec les arguments désirés.

### Execution :
```bash
python trainv2.py
```

## Structure du Code

- **ConvLSTMCell**  
  Implémente une cellule ConvLSTM qui concatène l'entrée avec l'état caché précédent, applique une convolution pour calculer les portes, et met à jour les états. Un contrôle d'erreur vérifie que les dimensions correspondent.

- **RegNetBlock**  
  Ce bloc combine des couches convolutionnelles et un module ConvLSTM avec une connexion résiduelle. Il gère le changement de dimensions via le paramètre de stride et réalise la fusion des caractéristiques via une convolution 1×1.

- **RegNetCIFAR & RegNetImageNet**  
  Ces classes construisent le réseau complet pour CIFAR-10 et ImageNet, respectivement, en empilant plusieurs RegNetBlocks. Dans RegNetCIFAR, pour chaque couche, les états caché et de la cellule sont réinitialisés pour éviter des conflits de dimensions.

- **Fonctions d'Entraînement et de Test**  
  La fonction `train_model` exécute la boucle d'entraînement en mettant à jour les poids du modèle et en affichant la progression (perte et précision). La fonction `test_model` évalue la précision du modèle sur le jeu de test.

- **Programme Principal**  
  Utilise `argparse` pour permettre la sélection du jeu de données et la configuration des hyperparamètres tels que le nombre d'époques, la taille du batch et le taux d'apprentissage. Selon le jeu de données choisi, les transformations appropriées et les DataLoaders sont créés.

## Remarques

- **Réinitialisation des États LSTM** :  
  Pour chaque couche du réseau, les états caché et de la cellule sont réinitialisés afin d'éviter des conflits de dimensions lors du passage entre les couches.

- **Gestion des Erreurs** :  
  La cellule ConvLSTM intègre une vérification des dimensions pour s'assurer que l'entrée et l'état caché sont compatibles, et renvoie une erreur descriptive si ce n'est pas le cas.

## Licence

Ce code est fourni à des fins pédagogiques. Vous êtes libre de le modifier et de l'utiliser selon vos besoins.
