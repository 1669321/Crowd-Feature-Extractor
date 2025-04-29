import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC

# 1. Cargar las imágenes de entrenamiento y entrenar k-means
def cargar_imagenes_y_entrenar_kmeans():
    all_descriptors = []  # Lista para guardar todos los descriptores
    for i in range(1, 11):  # Asumiendo que las imágenes son del 1 al 10
        image = cv2.imread(f"Where's Waldo Images/Raw/{i}.jpg")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)

    # Convertir todos los descriptores a un array numpy
    all_descriptors = np.vstack(all_descriptors)
    
    # Aplicar k-means para crear el vocabulario visual
    kmeans = KMeans(n_clusters=100, random_state=42)  # Ajusta el número de clusters (100 es un ejemplo)
    kmeans.fit(all_descriptors)
    
    return kmeans

# 2. Cargar y extraer el histograma de una imagen
def extraer_histograma_de_imagen(image, kmeans):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    histogram = np.zeros(len(kmeans.cluster_centers_))
    if descriptors is not None:
        for descriptor in descriptors:
            label = kmeans.predict([descriptor])
            histogram[label] += 1
    
    # Normalizar el histograma
    histogram = histogram / np.linalg.norm(histogram)
    return histogram

# 3. Entrenar el clasificador SVM
def entrenar_clasificador_svm(kmeans):
    # Cargar las imágenes de entrenamiento (en la carpeta "Cleared" y "Raw")
    X_train = []
    y_train = []
    
    for i in range(1, 11):  # Asumiendo que tienes las imágenes de entrenamiento en las carpetas correspondientes
        # Cargar imagen Raw (con Waldo)
        image_raw = cv2.imread(f"Where's Waldo Images/Raw/{i}.jpg")
        hist_raw = extraer_histograma_de_imagen(image_raw, kmeans)
        X_train.append(hist_raw)
        y_train.append(1)  # Etiqueta 1 para Raw (Waldo presente)
        
        # Cargar imagen Cleared (sin Waldo)
        image_cleared = cv2.imread(f"Where's Waldo Images/Cleared/{i}.jpg")
        hist_cleared = extraer_histograma_de_imagen(image_cleared, kmeans)
        X_train.append(hist_cleared)
        y_train.append(0)  # Etiqueta 0 para Cleared (Waldo no presente)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Entrenar el clasificador SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    
    return svm

# 4. Detectar a Waldo en la nueva imagen (imagen 11)
def detectar_waldo_en_imagen(image, kmeans, svm):
    window_size = (100, 100)  # Tamaño de la ventana deslizante (ajustar según sea necesario)
    step_size = 20  # Tamaño del paso en cada dirección de la ventana

    # Recorrer la imagen con la ventana deslizante
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            # Extraer la subimagen de la ventana deslizante
            window = image[y:y + window_size[1], x:x + window_size[0]]
            
            # Extraer el histograma de la ventana
            hist_window = extraer_histograma_de_imagen(window, kmeans)
            
            # Hacer la predicción con el clasificador SVM
            prediction = svm.predict([hist_window])
            
            # Si la predicción es positiva (Waldo detectado), dibujar un rectángulo alrededor de la ventana
            if prediction == 1:
                cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)

    # Mostrar la imagen con Waldo detectado
    cv2.imshow("Imagen con Waldo detectado", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Flujo principal del programa ---
# Paso 1: Entrenar el modelo k-means
kmeans = cargar_imagenes_y_entrenar_kmeans()

# Paso 2: Entrenar el clasificador SVM
svm = entrenar_clasificador_svm(kmeans)

# Paso 3: Cargar la imagen de prueba (imagen 11)
image_11 = cv2.imread("./Where's Waldo Images/Raw/11.jpg")

# Paso 4: Detectar a Waldo en la imagen de prueba
detectar_waldo_en_imagen(image_11, kmeans, svm)
