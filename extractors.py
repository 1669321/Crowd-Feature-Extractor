import cv2
import numpy as np

def extract_sift_features(image_paths):
    """
    Extrau els descriptors SIFT d'una llista d'imatges.
    :param image_paths: Llista de rutes d'imatges.
    :return: Un array de descriptors SIFT i un diccionari amb la ruta de cada imatge i els seus descriptors.
    """
    sift = cv2.SIFT_create()
    descriptors_list = []
    image_descriptor_map = []

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: No s'ha pogut llegir la imatge {path}.")
            continue
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            descriptors_list.extend(descriptors)
            image_descriptor_map.append((path, descriptors))
        else:
            print(f"No descriptors trobats per la imatge {path}.")
    return np.array(descriptors_list), image_descriptor_map

def get_image_histogram(descriptors, kmeans, k):
    """
    Calcula l'histograma de paraules visuals per una imatge a partir dels descriptors.
    :param descriptors: Descriptors SIFT de la imatge.
    :param kmeans: Model KMeans ajustat per el diccionari visual.
    :param k: Nombre de clústers (paraules visuals).
    :return: Histograma de paraules visuals.
    """
    hist = np.zeros(k)
    if descriptors is not None:
        clusters = kmeans.predict(descriptors)
        for c in clusters:
            hist[c] += 1
    return hist