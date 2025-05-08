import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from extractors import get_image_histogram

def sliding_window_predict(image_path, kmeans, clf, k, window_size=64, step=16):
    """
    Realitza la predicció sobre una imatge utilitzant una finestra lliscant.
    :param image_path: Ruta de la imatge a processar.
    :param kmeans: Model KMeans entrenat pel diccionari visual.
    :param clf: Classificador entrenat (SVM).
    :param k: Nombre de clústers del model KMeans.
    :param window_size: Mida de la finestra lliscant.
    :param step: Pas de la finestra lliscant.
    """
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    detected = []

    for y in range(0, img.shape[0] - window_size, step):
        for x in range(0, img.shape[1] - window_size, step):
            crop = img[y:y + window_size, x:x + window_size]
            _, descriptors = sift.detectAndCompute(crop, None)
            hist = get_image_histogram(descriptors, kmeans, k).reshape(1, -1)
            prob = clf.predict_proba(hist)[0][1]
            if prob > 0.8:
                detected.append((x, y, prob))

    for x, y, p in detected:
        cv2.rectangle(img, (x, y), (x+window_size, y+window_size), (0,0,255), 2)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Detecció de Wally")
    plt.axis('off')
    plt.show()

def detect_in_folder(model_path="bovw_waldo_model.pkl", test_path="Where's Waldo Images/Raw/Test/"):
    """
    Detecta a Wally en totes les imatges d'una carpeta.
    :param model_path: Ruta del model entrenat (KMeans + SVM).
    :param test_path: Ruta de la carpeta amb les imatges a processar.
    """
    kmeans, clf = joblib.load(model_path)
    for test_img in os.listdir(test_path):
        test_img_path = os.path.join(test_path, test_img)
        print(f"Processant {test_img}...")
        sliding_window_predict(test_img_path, kmeans, clf, kmeans.n_clusters)

if __name__ == "__main__":
    detect_in_folder()