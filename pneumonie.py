# Import des modules nécessaires de keras et tensorflow pour la realisation du CNN
from keras import layers
from keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import matplotlib.pyplot as plt

# on utilise la data augmentation pour reduire l'overfitting de notre modele
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    shear_range=0.05,  
    zoom_range=0.1,  
    horizontal_flip=False, 
    fill_mode='nearest')  

#  On normalise les 3 jeux de donnees

test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# ici on definit notre ensemble d apprentissage
# donc qund on va faire l apprentissa, on le fait sur train_generator
train_generator = train_datagen.flow_from_directory(
        '/Users/domin/Downloads/archive/chest_xray/train',
        # met toutes les images de la meme taille, c est pas trop petit pour ne pas ecraser et   
        target_size=(64, 64),  
        #taille des images chargees pendant l apprentissage
        batch_size=32,  
        #probleme de classification binaire
        class_mode='binary')  

# on fait la meme chose pour la validation
validation_generator = val_datagen.flow_from_directory(
        '/Users/domin/Downloads/archive/chest_xray/val',  
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')
# on utilise les donnees de test
test_generator = test_datagen.flow_from_directory(
        '/Users/domin/Downloads/archive/chest_xray/test',  
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# codage du modele avec les differentes couches de convolution pour trouver les features puis
# une couche flatten pour vectoriser la sortie, vecteur qui sera utilise pour l'entrainement 
#du modele de classification binaire
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D(2, 2))    
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())  
model.add(layers.Dense(128, activation='relu'))  
model.add(layers.Dense(1, activation='sigmoid'))  

#Probleme binaire donc binary cross entropy
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# On affiche les informations sur le modele
model.summary()

for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# On lance l'apprentissage, il va chercher dans train generator, le step par epochs, le nombre d epochs
history = model.fit(
      train_generator,
      steps_per_epoch=100,  
      epochs=20,  
      validation_data=validation_generator,
      validation_steps=50)  

# on teste le modele sur les donnees de test
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)

print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# on sauvegarde le modele pre entraine
model.save('pneumonie.h5')

# On recupere les resultats du modele pour les afficher
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# on affiche des graphiques des resultats des donnees d'entrainement et de validation
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()