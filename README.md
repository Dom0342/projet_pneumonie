# Projet_pneumonie
# Ce projet est un projet de Deep Learning utilisant un reseau de neurones de convolution (CNN) permettant une classfication binaire d'images de radio de poumons possedant ou non une pneummonie
## Ce projet utilise keras/tensorflow pour la creation du CNN
### J'ai telecharge une base donnees deja partitionnee entre donnees d'entrainement et de validation.
### Pour ameliorer la qualite du modele et eviter l'overfitting, utilisation de data augmentation en appliquant des operations sur les images pour augmenter la variete des images d'entree et reduire la variance du modele, normalisation des donnees ( diviser les pixels par 255).
### Creation d'un modele CNN avec plusieurs couches de convolution utilisant une fonction d'activation ReLU pour extraire les features des images, puis des couches de maxPooling pour reduire la taille des donnees et le nombre de parametres a entrainer tout en gardant la pertinence des donnees, ajout d'une couche Flatten pour vectoriser la sortie du reseau et l'appliquer a une couche Dense pour la classification finale.
### Compilation et entrainement du modele sur le nombre d'epochs et de steps par epochs pour  entrainer toutes les images de la base de donnees, entrainement avec une fonctions loss "binary-cross-entropy" optimiseur Adam et mesure de la performance avec accuracy.
### Evaluation du modele sur les donnees de test et obtient une accuracy de 85%.
