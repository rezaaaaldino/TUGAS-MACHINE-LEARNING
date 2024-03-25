import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage import io, transform
import numpy as np

# Ganti dengan direktori yang berisi folder dataset
dataset_dir = 'D:\SEMESTER 4\TUGAS ML\Dataset'

# Inisialisasi daftar kelas dan label
classes = ['cat', 'dog', 'rabbit',]
labels = [0, 1, 2]

X_train = []
y_train = []

# Load data training
for label, class_name in zip(labels, classes):
    class_dir = os.path.join(dataset_dir, class_name)
    
    # Untuk setiap gambar dalam kelas
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        
        # Ubah gambar menjadi ukuran yang sama
        image = io.imread(image_path, as_gray=True)
        image = transform.resize(image, (100, 100))  # Ubah ukuran gambar menjadi 100x100
        
        # Ubah gambar menjadi larik nilai piksel
        image_flattened = image.flatten()
        
        # Tambahkan gambar dan label kelas ke data training
        X_train.append(image_flattened)
        y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

print("Data Training (X_train):")
print(X_train)
print("Label Training (y_train):")
print(y_train)

# Inisialisasi model SVM
svm = SVC()

# Latih model SVM menggunakan data training
svm.fit(X_train, y_train)

# Inisialisasi model KNN
knn = KNeighborsClassifier()

# Latih model KNN menggunakan data training
knn.fit(X_train, y_train)

# Load data testing
X_test = []
y_test = []
test_dir = 'D:\SEMESTER 4\TUGAS ML\dataset_test'

# Untuk setiap gambar dalam folder testing
for image_name in os.listdir(test_dir):
    image_path = os.path.join(test_dir, image_name)
    
    # Ubah gambar menjadi ukuran yang sama
    image = io.imread(image_path, as_gray=True)
    image = transform.resize(image, (100, 100))  # Ubah ukuran gambar menjadi 100x100
    
    # Ubah gambar menjadi larik nilai piksel
    image_flattened = image.flatten()
    
    # Tambahkan gambar dan label kelas ke data testing
    X_test.append(image_flattened)
    # Tambahkan label dummy karena tidak diketahui
    y_test.append(-1)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Prediksi kelas dari data testing menggunakan model SVM
y_pred_svm = svm.predict(X_test)

# Cetak hasil prediksi menggunakan model SVM
print("Hasil Prediksi (SVM):")
for i, pred in enumerate(y_pred_svm):
    class_name = classes[pred]
    print("Gambar", i+1, ": Prediksi =", class_name)

# Prediksi kelas dari data testing menggunakan model KNN
y_pred_knn = knn.predict(X_test)

# Cetak hasil prediksi menggunakan model KNN
print("Hasil Prediksi (KNN):")
for i, pred in enumerate(y_pred_knn):
    class_name = classes[pred]
    print("Gambar", i+1, ": Prediksi= ", class_name)

# Evaluasi akurasi model SVM
y_train_pred_svm = svm.predict(X_train)
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print("Akurasi Model SVM (Data Training):", train_accuracy_svm)

# Evaluasi akurasi model KNN
y_train_pred_knn = knn.predict(X_train)
train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)
print("Akurasi Model KNN (Data Training):", train_accuracy_knn)