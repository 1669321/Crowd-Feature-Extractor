import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from glob import glob

def load_images_from_folder(folder):
    images = []
    print(f"Carregant imatges de la carpeta {folder}...")
    for filename in glob(os.path.join(folder, '*.jpg')):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    print(f"S'han carregat {len(images)} imatges de la carpeta {folder}.")
    return images

def extract_sift_features(images):
    sift = cv2.SIFT_create(nfeatures=1000)
    descriptors = []
    print("Extraient característiques SIFT...")
    for idx, image in enumerate(images):
        keypoints, desc = sift.detectAndCompute(image, None)
        if desc is not None:
            descriptors.append(desc)
        if (idx + 1) % 10 == 0:
            print(f"Processades {idx + 1} imatges amb SIFT.")
    print(f"S'han extren característiques SIFT de {len(images)} imatges.")
    return descriptors

def build_bow(descriptors, k=100):
    all_descriptors = np.vstack(descriptors)
    print(f"Construint el model Bag of Visual Words amb {len(all_descriptors)} descriptors...")
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(all_descriptors)
    print(f"Model BoVW construit amb {k} clústers.")
    return kmeans

def image_to_bow_histogram(image, kmeans):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        clusters = kmeans.predict(descriptors)
        histogram = np.histogram(clusters, bins=np.arange(kmeans.n_clusters + 1), density=True)[0]
        return histogram
    else:
        return np.zeros(kmeans.n_clusters)

def detect_waldo(image, kmeans, knn, window_size=(64, 64), step_size=32):
    h, w = image.shape
    detections = []

    for y in range(0, h - window_size[1], step_size):
        for x in range(0, w - window_size[0], step_size):
            sub_image = image[y:y + window_size[1], x:x + window_size[0]]

            sub_histogram = image_to_bow_histogram(sub_image, kmeans)

            pred = knn.predict([sub_histogram])

            if pred == 1:
                detections.append((x, y, window_size[0], window_size[1]))
                cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 0), 2)
    return image, detections

print("Carregant imatges d'entrenament...")
train_images = load_images_from_folder("Where's Waldo Images/Raw/Train")
train_labels = [1] * len(train_images)
                            
print("Carregant imatges sense Wally...")
cleared_images = load_images_from_folder("Where's Waldo Images/Cleared")
train_images.extend(cleared_images)
train_labels.extend([0] * len(cleared_images))

descriptors = extract_sift_features(train_images)

kmeans = build_bow(descriptors, k=100)

print("Convertint imatges en histogrames BoVW...")
train_histograms = [image_to_bow_histogram(img, kmeans) for img in train_images]

print("Dividint el conjunt d'entrenament en entrenament i validació...")
X_train, X_val, y_train, y_test = train_test_split(train_histograms, train_labels, test_size=0.3, random_state=42)

print("Entrenant el classificador KNN...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

print("Validant el model...")
y_pred = knn.predict(X_val)
print(classification_report(y_test, y_pred))

print("Testejant en noves imatges...")
test_images = load_images_from_folder("Where's Waldo Images/Raw/Test")

for i, img in enumerate(test_images):
    if i == 2:
        img_copy = img.copy()
        marked_image, detections = detect_waldo(img_copy, kmeans, knn)
        
        cv2.imshow(f"Imatge {i + 1} amb deteccions", marked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"S'han trobat {len(detections)} possibles ubicacions de Wally a la imatge {i+1}.")