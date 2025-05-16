# Visualization
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time 

import cv2
import numpy as np
import os
import model
from model import get_conv

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directoris de les imatges (Amb Waldo i Sense Waldo)
waldo_dir = "Wheres Waldo Images/Waldo"
notwaldo_dir = "Wheres Waldo Images/NotWaldo"

def plot_image(i, predictions_array, true_label, img):
    """
    Funció per mostrar una imatge amb la seva predicció i la seva etiqueta real.
    :param i: Índex de la imatge a mostrar.
    :param predictions_array: Array de prediccions del model.
    :param true_label: Etiqueta real de la imatge.
    :param img: Imatge original.
    :return: None.
    """
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    print(int(true_label), predicted_label, predictions_array, 100 * np.max(predictions_array))
    plt.xlabel(f"{class_names[predicted_label]} {100 * np.max(predictions_array):2.0f}% ({class_names[int(true_label)]})", color=color)


def plot_value_array(i, predictions_array, true_label):
    """
    Funció per mostrar un gràfic de barres de les probabilitats predites per a una imatge.
    :param i: Índex de la imatge a mostrar.
    :param predictions_array: Array de probabilitats predites del model.
    :param true_label: Etiqueta real de la imatge.
    :return: None.
    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[int(true_label)].set_color('blue')


def locate():
    """
    Funció per localitzar Waldo en una imatge gran utilitzant sliding window i generant un heatmap.
    :return: Imatge amb la localització de Waldo marcada."""
    
    # Carregar la imatge gran on buscarem a Waldo
    data = cv2.cvtColor(cv2.imread("Wheres Waldo Images/Raw/Test/8.jpg"), cv2.COLOR_BGR2RGB)

    # Parametres de la finestra lliscant
    window_size = 64
    stride = 8
    h, w, _ = data.shape
    heatmap = np.zeros((h // stride, w // stride)) # Mapa de calor buit

    # Recorrem la imatge amb finestres 64x64
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            patch = data[y:y + window_size, x:x + window_size]
            patch_input = patch.reshape(1, 64, 64, 3)
            patch_input = patch_input.astype('float32')
            pred = heatmodel.predict(patch_input, verbose=0)
            heatmap[y // stride, x // stride] = pred[0][0]

    # Mostrar el mapa de calor amb probabilitats
    plt.imshow(heatmap, cmap='hot')
    plt.title("Heatmap de probabilitat")
    plt.colorbar()
    plt.show()

    # Mostrar la màscara on la probabilitat és superior a 0.7
    mask = heatmap > 0.7
    plt.imshow(mask, cmap='gray')
    plt.title("Area de detecció")
    plt.show()

    # Dibuixa un rectangle a la posició de Waldo
    annotated = data.copy()
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:
                cv2.rectangle(annotated, (x * stride, y * stride), (x * stride + window_size, y * stride + window_size), (0, 0, 255), 2)

    return annotated


X, Y = [], []

# Carrega les imatges amb Waldo i els assigna etiqueta 1
for img_name in os.listdir(waldo_dir):
    img_path = os.path.join(waldo_dir, img_name)
    img = Image.open(img_path)
    img = img.resize((64, 64)) 
    img_array = np.array(img)
    X.append(img_array)
    Y.append(1)

# Carrega les imatges sense Waldo i els assigna etiqueta 0
for img_name in os.listdir(notwaldo_dir):
    img_path = os.path.join(notwaldo_dir, img_name)
    img = Image.open(img_path)
    img = img.resize((64, 64))  
    img_array = np.array(img)
    X.append(img_array)
    Y.append(0)  

# Conversió a arrays numpy
X = np.array(X)
Y = np.array(Y)

# Divisió de les dades en conjunts d'entrenament i prova (90% / 10%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)


print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Crear un generador que aplica transformacions aleatòries a les imatges per millorar l'entrenament
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(X_train)

# Creació i entrenament del model
model = model.get_conv(input_shape=(64, 64, 3))

# Entrena el model amb augmentació de dades
model.fit(datagen.flow(X_train, Y_train, batch_size=32), epochs=15, verbose=1, validation_data=(X_test, Y_test))

# Avalua el model amb les dades de test
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# Guarda els pesos del model
model.save_weights("localize4.weights.h5")

# Fa la predicció sobre les imatges de test
predictions = model.predict(X_test)
threshold = 0.7 # Umbral per considerar que hi ha Waldo

# Mostra les primeres 20 imatges de test amb les seves prediccions
plt.figure(figsize=(10, 10))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(X_test[i])
    predicted_label = 1 if predictions[i] > threshold else 0
    color = 'blue' if predicted_label == Y_test[i] else 'red'
    plt.title(f"Pred: {predicted_label}, True: {Y_test[i]}", color=color)
    plt.xticks([]), plt.yticks([])
plt.show()

# Predicció individual per una sola imatge
img = X_test[0]
img = np.expand_dims(img, 0)


class_names = ['Not Waldo', 'Waldo']
predictions_single = model.predict(img)
plot_value_array(0, predictions_single, Y_test)
_ = plt.xticks(range(len(class_names)), class_names, rotation=45)

# Carrega el model per aplicar-lo sobre imatges grans (amb els pesos entrenats)
heatmodel = get_conv(input_shape=(64, 64, 3), filename="localize4.weights.h5")

# Mesura el temps d'execució de la funció locate
start = time.time()
annotated = locate()
end = time.time()
print(f"Temps d'execució de locate(): {end - start:.2f} segons")

# Mostra la imatge amb el rectangle dibuixat on s'ha localitzat Waldo
plt.title("Augmented")
plt.imshow(annotated)
plt.show()