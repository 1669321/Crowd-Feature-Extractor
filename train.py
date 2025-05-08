import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import SVC
import joblib
from extractors import extract_sift_features, get_image_histogram

def load_image_paths(folder):
    """
    Carrega totes les rutes de les imatges dins d'una carpeta
    :param folder: carpeta que conté les imatges
    :return: llista de rutes d'imatges
    """
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]

def build_vocabulary(descriptors, k=100):
    """
    Construeix el diccionari visual utilitzant KMeans
    :param descriptors: descriptors SIFT de les imatges
    :param k: nombre de clústers per al KMeans
    :return: model KMeans ajustat
    """
    minibatch_kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000, random_state=42)
    minibatch_kmeans.fit(descriptors)
    return minibatch_kmeans

def train_model(waldo_path, not_waldo_path, model_output_path="bovw_waldo_model.pkl", k=100):
    """
    Entrena el model utilitzant imatges de Wally i no-Wally
    :param waldo_path: carpeta amb imatges de Wally
    :param not_waldo_path: carpeta amb imatges sense Wally
    :param model_output_path: ruta on guardar el model entrenat
    :param k: nombre de clústers per al KMeans
    """
    waldo_imgs = load_image_paths(waldo_path)
    not_waldo_imgs = load_image_paths(not_waldo_path)
    all_imgs = waldo_imgs + not_waldo_imgs
    labels = [1] * len(waldo_imgs) + [0] * len(not_waldo_imgs)

    print("Extraient descriptors SIFT...")
    descriptors, descriptor_map = extract_sift_features(all_imgs)

    print("Entrenant KMeans...")
    kmeans = build_vocabulary(descriptors, k=k)

    print("Construint histogrames...")
    X = []
    for path, desc in descriptor_map:
        hist = get_image_histogram(desc, kmeans, k)
        X.append(hist)

    X = np.array(X)
    y = np.array(labels)

    print("Entrenant SVM...")
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y)

    joblib.dump((kmeans, clf), model_output_path)
    print(f"Model guardat en {model_output_path}")

if __name__ == "__main__":
    train_model("Where's Waldo Images/Raw/Train", "Where's Waldo Images/Cleared")
    