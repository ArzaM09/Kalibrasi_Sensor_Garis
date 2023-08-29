#import library
import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('/home/blank/coding/python/ML_project/kalibrasi_otomatis/data.csv')

 
#Melakukan Klasifikasi Warna Menggunakan Algortima KNN 
training_data = data #mendefenisikan data diatas menjadi training_data

def predict_knn (training_data, data_sensor, K): #K adalah tetangga terdekat
  # Menghitung batas atas dan batas bawah dari data sensor 
    mean = np.mean(data_sensor)
    std_dev = np.std(data_sensor)
    k=1
    batas_atas = mean + k * std_dev
    batas_bawah = mean - k * std_dev
  # For diatas ini merupakan kegiatan memisahkan fitur dan label dengan cara pengecekan melalui batas atas dan bawah 
    pembagian_data = training_data[(training_data['Data '] >= batas_bawah) & (training_data['Data '] <= batas_atas)]
    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(pembagian_data[['Data ']], pembagian_data['Label'])
    
    hasil = []
    for warna in data_sensor:
        if batas_bawah <= warna <= batas_atas:
            predicted_class = classifier.predict([[warna]])
            hasil.append(predicted_class[0])
        else:
            hasil.append("Tidak Ada")
    return hasil

data_sensor = [69,67,68,67,68,67,68,54] #data Percobaan
hasil = predict_knn(training_data, data_sensor, K=3)
print("Hasil Prediksi Warna : ",hasil[0])