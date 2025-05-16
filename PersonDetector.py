import os
import cv2
import numpy as np
import joblib

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

def load_paired_images(train_folder, cleared_folder):
    """Carrega pars d'imatges a color des d'una carpeta"""
    train_images = sorted(os.listdir(train_folder))
    cleared_images = sorted(os.listdir(cleared_folder))
    pairs = []
    for t, c in zip(train_images, cleared_images):
        if t == c:
            img_with = cv2.imread(os.path.join(train_folder, t), cv2.IMREAD_COLOR)
            img_cleared = cv2.imread(os.path.join(cleared_folder, c), cv2.IMREAD_COLOR)
            if img_with is not None and img_cleared is not None:
                pairs.append((img_with, img_cleared))
    return pairs

def extract_window_descriptors(pairs, window_size=(16, 16), step_size=64, max_features=1000):
    """Extrau descriptors SIFT de finestres positives i negatives fent servir diferencies entre imatges"""
    sift = cv2.SIFT_create(nfeatures=max_features)
    X_pos, X_neg = [], []

    for img_with, img_cleared in tqdm(pairs, desc="Processant imatges"):
        for y in range(0, img_with.shape[0] - window_size[1], step_size):
            for x in range(0, img_with.shape[1] - window_size[0], step_size):
                win_with = img_with[y:y+window_size[1], x:x+window_size[0]]
                win_clear = img_cleared[y:y+window_size[1], x:x+window_size[0]]

                diff = cv2.absdiff(win_with, win_clear)
                diff_score = np.mean(diff)

                gray = cv2.cvtColor(win_with, cv2.COLOR_BGR2GRAY)
                kp, des = sift.detectAndCompute(gray, None)
                if des is None:
                    continue

                if diff_score > 25:
                    X_pos.append(des)
                else:
                    X_neg.append(des)
    return X_pos, X_neg

def extract_sift_descriptors(images, max_features=500):
    """Extrau descriptors SIFT d'una llista d'imatges"""
    sift = cv2.SIFT_create(nfeatures=max_features)
    descriptors_list = []
    for img in tqdm(images, desc="Extraient carecterístiques SIFT"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    return descriptors_list

def create_bovw_vocabulary(descriptors, vocab_size=100):
    """Crea un vocabulari visual agrupant descriptors amb KMeans"""
    kmeans = KMeans(n_clusters=vocab_size, random_state=42, verbose=1, n_init=10)
    kmeans.fit(descriptors)
    return kmeans

def compute_bovw_histograms(descriptor_list, kmeans_model, vocab_size):
    """Converteix descriptors en histogrames de paraules visuals"""
    histograms = []
    for descriptors in tqdm(descriptor_list, desc="Computant histogrames"):
        if descriptors is not None and len(descriptors) > 0:
            words = kmeans_model.predict(descriptors)
            hist, _ = np.histogram(words, bins=np.arange(vocab_size + 1))
            hist = hist.astype(float) / np.sum(hist)
        else:
            hist = np.zeros(vocab_size)
        histograms.append(hist)
    return np.array(histograms)

def sliding_window(image, step_size, window_size):
    """Generador de finestres delliscants sobre una imatge"""
    for y in range(0, image.shape[0] - window_size[1], step_size):
       for x in range(0, image.shape[1] - window_size[0], step_size):
           yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect_waldo_in_image(img, sift, kmeans, svm, vocab_size, window_size=(16, 16), step_size=32):
    """Aplica sliding window i predicció SVM per trobar a Wally"""
    max_score = 0
    best_window = None
    
    for (x, y, window) in sliding_window(img, step_size, window_size):
        if window.shape[:2] != (window_size[1], window_size[0]):
            continue
        
        gray = cv2.cvtColor(window, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        if des is None:
            continue
        words = kmeans.predict(des)
        hist, _ = np.histogram(words, bins=np.arange(vocab_size + 1))
        hist = hist.astype(float) / np.sum(hist)
        hist = hist.reshape(1, -1)
        score = svm.predict_proba(hist)[0][1]
        if score > max_score:
            max_score = score
            best_window = (x, y, x + window_size[0], y + window_size[1])
            
    return best_window, max_score

# Extraiem exemples positius i negatius realistes
train_folder = "Where's Waldo Images/Raw/Train"
cleared_folder = "Where's Waldo Images/Cleared"
pairs = load_paired_images(train_folder, cleared_folder)[:5]
X_pos_desc, X_neg_desc = extract_window_descriptors(pairs)
 
# Generació del diccionari visual
vocab_size = 100
all_descriptors = np.vstack(X_pos_desc + X_neg_desc)
kmeans = KMeans(n_clusters=vocab_size, random_state=42, verbose=1, n_init=10)
kmeans.fit(all_descriptors)

# Convertim cada imatge en un histograma
X_pos = compute_bovw_histograms(X_pos_desc, kmeans, vocab_size)
X_neg = compute_bovw_histograms(X_neg_desc, kmeans, vocab_size)

# Creem les etiquetes
y_pos = np.ones(len(X_pos)) # 1 per Wally
y_neg = np.zeros(len(X_neg)) # 0 per no-Wally

# Dataset complert
X = np.vstack((X_pos, X_neg))
y = np.concatenate((y_pos, y_neg))

# Entrenament del classificador
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluació
y_pred = svm.predict(X_val)
print("Accuracy en validació:", accuracy_score(y_val, y_pred))
print("\nReport de classificació:\n", classification_report(y_val, y_pred))

# Carregar imatges de test
test_images = sorted(os.listdir("Where's Waldo Images/Raw/Test"))
sift = cv2.SIFT_create()

# Buscar a Wally a cada imatge
for idx, filename in enumerate(test_images):
    path = os.path.join("Where's Waldo Images/Raw/Test", filename)
    img = cv2.imread(path)
    box, score = detect_waldo_in_image(img, sift, kmeans, svm, vocab_size)
    if box:
        x1, y1, x2, y2 = box
        output = img.copy()
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, f"Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.imshow(f"Detected Waldo - {idx}", output)
        cv2.waitKey(0)
cv2.destroyAllWindows()