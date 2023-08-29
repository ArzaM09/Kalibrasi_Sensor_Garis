#import library
import pandas as pd 
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier

# Menentukan Batas Atas dan Batas Bawah Setiap Data 
#Mengambil data putih, abu-abu, dan hitam
data = pd.read_csv('/home/blank/coding/python/ML_project/kalibrasi_otomatis/data.csv')
datas = pd.DataFrame(data)
#merubah data yang diambil menjadi array 
data_putih = data['Data '].iloc[0:121].values
data_abu = data['Data '].iloc[122:242].values
data_hitam = data['Data '].iloc[242:363].values


#Fungsi Menghitung batas atas dan bawah
def batas (data, k):
  mean = np.mean(data)
  std_dev = np.std(data)
    
  batas_atas = mean + k * std_dev
  batas_bawah = mean - k * std_dev
    
  return batas_atas, batas_bawah

#Menghitung Batas Atas dan Batas Bawah Setiap warna
k = 1 #nilai rentang data
batas_atasp, batas_bawahp = batas(data_putih,k) #data putih
batas_atasa, batas_bawaha = batas(data_abu,k) #data abu-abu
batas_atash, batas_bawahh = batas(data_hitam,k) #data hitam


#Melakukan Klasifikasi Warna Menggunakan Algortima KNN 
training_data = data #mendefenisikan data diatas menjadi training_data

def predict_knn (training_data, data_sensor, K): #K adalah tetangga terdekat

  # For diatas ini merupakan kegiatan memisahkan fitur dan label dengan cara pengecekan melalui batas atas dan bawah 
    pembagian_data = training_data[(training_data['Data '] >= batas_bawahp) & (training_data['Data '] <= batas_atash)]
    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(pembagian_data[['Data ']], pembagian_data['Label'])
    
    hasil = []
    for warna in data_sensor:
        if batas_bawahp <= warna <= batas_atash:
            predicted_class = classifier.predict([[warna]])
            hasil.append(predicted_class[0])
        else:
            hasil.append("Tidak Ada")
    return hasil

data_sensor = [69, 730, 890, 62.3, 750, 68] #data Percobaan
hasil = predict_knn(training_data, data_sensor, K=3)
for i, result in enumerate(hasil):
    print("Warna baru ke-{} diklasifikasikan sebagai: {}".format(i+1, result))
