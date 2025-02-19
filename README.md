# RegNet_Proj


---

<div align = center>
  
&nbsp; ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
&nbsp; ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
&nbsp; ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white)
&nbsp; ![Deep Learning](https://img.shields.io/badge/Deep%20Learning-000000?logo=deep-learning&logoColor=white)
&nbsp; ![RegNet](https://img.shields.io/badge/RegNet-008080?logo=github&logoColor=white)
&nbsp; ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

---


[Résumé](#résumé-de-l'article-et-contribution) | [Analyse](#analyse-de-la-méthode) | [Tests et évaluation](#tests-et-évaluation) | [Conclusion](#conclusion-générale)

</div>

### Résumé de l'article et contribution

L'article propose **RegNet**, une architecture basée sur **ResNet** qui introduit un **module régulateur** utilisant des **RNN convolutionnels** (ConvRNNs, notamment ConvLSTM et ConvGRU). L'objectif est de **mieux exploiter les informations spatiales et temporelles** dans les réseaux résiduels, en contournant les limitations des connexions de raccourci classiques. Expérimenté sur **CIFAR-10, CIFAR-100 et ImageNet**, RegNet **améliore les performances** des modèles ResNet et SE-ResNet tout en nécessitant un nombre limité de paramètres supplémentaires.

### Analyse de la méthode

#### **Données utilisées**  
L’évaluation de RegNet a été réalisée sur trois jeux de données bien connus en classification d'images :  
- **CIFAR-10** (10 classes, 50K images d’entraînement, 10K de test)  
- **CIFAR-100** (100 classes, même structure que CIFAR-10)  
- **ImageNet** (1.28M images d’entraînement, 50K de validation, 1000 classes)

#### **Modèle et architecture**  
RegNet repose sur ResNet en y ajoutant un **module régulateur** sous la forme d’un **ConvRNN** inséré entre les blocs résiduels. Il existe deux variantes principales :
- **RegNet classique** : Ajout d’un ConvRNN dans chaque bloc résiduel.
- **Bottleneck RegNet** : Optimisation pour les modèles plus profonds, utilisant une architecture en goulot d’étranglement.

#### **Optimisation et entraînement**  
Les modèles sont entraînés avec **SGD** (momentum 0.9, weight decay 1e-4), et une **réduction progressive du taux d’apprentissage**.  
Des techniques classiques d’**augmentation de données** (recadrage, miroir, etc.) sont utilisées.

#### **Résultats obtenus**  
- **CIFAR-10/100** : RegNet réduit significativement l’erreur par rapport à ResNet et SE-ResNet.  
- **ImageNet** : RegNet-50 surpasse ResNet-50 et se rapproche des performances de ResNet-101 avec moins de calculs.  
- **Meilleure efficacité paramétrique** : RegNet atteint de meilleures performances avec **moins de couches** qu’un ResNet classique.

---

### **Tests et évaluation**  

#### **Objectif des tests**  
Évaluer la robustesse de RegNet sur la **classification d'images**, en comparant ses performances avec celles de ResNet et SE-ResNet.  
Les critères testés :
- **Précision du modèle** (Top-1 et Top-5 accuracy)  
- **Impact du module régulateur selon son positionnement**  
- **Efficacité paramétrique** (gain en performance vs. coût en calculs)

- #### **Données utilisées**  
- **CIFAR-10/100** pour des tests sur de petites images  
- **ImageNet** pour des images plus complexes et en haute résolution  

#### **Observations**  
- RegNet améliore la classification par rapport aux modèles classiques, grâce à une meilleure exploitation des dépendances spatiales et temporelles.  
- L’ajout du module ConvRNN est **plus efficace sur les couches basses** du réseau.  
- RegNet permet de **réduire la profondeur** nécessaire pour atteindre une précision donnée.

#### **Problèmes rencontrés**  
- L’augmentation de la taille du modèle rend l'entraînement **plus coûteux en calcul**.  
- Un ajustement fin des **hyperparamètres** (ex. taux d’apprentissage) est nécessaire pour éviter le surajustement.

---

### **Conclusion générale**  
RegNet constitue une **amélioration significative** de ResNet en exploitant un module régulateur basé sur des **ConvRNNs**. Il permet d’**apprendre des caractéristiques complémentaires** et **d’améliorer la classification des images** tout en **réduisant le besoin de profondeur du réseau**. Ces résultats ouvrent la voie à son application dans d'autres architectures basées sur ResNet et d'autres tâches comme la détection d’objets et la super-résolution.

## Branche de rendu 
Master

## Techniciens
<a href = "https://codefirst.iut.uca.fr/git/ouriahi">
<img src ="https://codefirst.iut.uca.fr/git/avatars/84062b2bb326d9e9154a9859b375e599?size=870" height="50px">
</a>
<a href = "https://codefirst.iut.uca.fr/git/nathan.verdier">
<img src ="https://codefirst.iut.uca.fr/git/avatars/84062b2bb326d9e9154a9859b375e599?size=870" height="50px">
</a>


<div align = center>
</div>
