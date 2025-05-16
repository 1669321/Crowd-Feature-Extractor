from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Lambda, Dense
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def get_conv(input_shape=(64, 64, 3), filename=None):
    """Funció que construeix i retorna un model convolucional per classificar imatges.
    :param input_shape: Tupla que indica la forma d'entrada de les imatges (default: 64x64x3).
    :param filename: Ruta opcional d'un arxiu de pesos .h5 per carregar els pesos en el model.
    :return: Un model keras compilat.
    """
    model = Sequential() # Inicialitzem el model com a seqüencial

    # Capa de normalització: converteix els valors de píxels de [0, 255] a [-1, 1]
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))

    # Primera capa convolucional: 32 filtres de 3x3, activació ReLu
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Redueix a la meitat la mida espacial de la imatge (64x64 -> 32x32)

    # Segona capa convolucional: 64 filtres
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 32x32 -> 16x16

    # Tercera capa convolucional: 128 filtres
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 16x16 -> 8x8

    # Quarta capa convolucional: 128 filtres
    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2))) # 8x8 -> 4x4

    # Capa de regularització: apaga el 50% de les neurones aleatòriament per evitar l'overfitting
    model.add(Dropout(0.5))

    # Aplana la sortida 3D a un vector 1D per connectar amb la capa densa final
    model.add(Flatten())

    # Capa de sortida amb 1 neurona i activació 'sigmoid' per classificar entre dues classes (Waldo o No-Waldo)
    model.add(Dense(1, activation="sigmoid"))
  
    # Compilació del model: usa binary_crossentropy com a funció de pèrdua, Adam com a optimitzador i 'accuracy' com a mètrica
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Si es proporciona un nom d'arxiu de pesos, carrega els pesos del model
    if filename:
        model.load_weights(filename)

    #Mostra el resum del model 
    model.summary()
    return model