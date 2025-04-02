import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Modeli yükle
model = load_model('heart_disease_model.h5')

# Kullanıcıdan veri girişi al
st.title('Kalp Hastalığı Tahmin Uygulaması')

st.write("""
    Bu uygulama, kalp hastalığı olup olmadığını tahmin etmek için
    kullanıcıdan gelen verileri kullanarak sonucu gösterecektir.
""")

# Kullanıcıdan giriş almak için form
age = st.slider("Yaş", 20, 80, 50)
sex = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
cp = st.selectbox("Göğüs Ağrısı Tipi", [0, 1, 2, 3])  # Örnek: 0 - Tip 1, 1 - Tip 2, vb.
trestbps = st.slider("Dinlenme Kan Basıncı", 90, 200, 120)
chol = st.slider("Serum Kolesterol", 100, 400, 200)
fbs = st.selectbox("Açlık Kan Şekeri", [0, 1])  # 0: < 120 mg/dl, 1: >= 120 mg/dl
restecg = st.selectbox("Elektrokardiyogram Sonuçları", [0, 1, 2])  # 0, 1, 2 değerleri
thalach = st.slider("Maksimum Kalp Atış Hızı", 60, 200, 150)
exang = st.selectbox("Egzersizle Ağrı", [0, 1])  # 0: Evet, 1: Hayır
oldpeak = st.slider("Depresyon", 0.0, 6.0, 1.0)
slope = st.selectbox("Egzersiz Pik Yüksekliği", [0, 1, 2])
ca = st.selectbox("Vasküler Hastalık Sayısı", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia Durumu", [3, 6, 7])

# Girdi verilerini DataFrame'e dönüştür
user_input = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == "Erkek" else 0],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Veriyi ölçeklendirme (modelinizin daha iyi çalışması için)
scaler = StandardScaler()
user_input_scaled = scaler.fit_transform(user_input)

# Tahmin yap
prediction = model.predict(user_input_scaled)
prediction = np.argmax(prediction, axis=1)

# Sonuçları göster
if prediction == 0:
    st.write("Sonuç: Kalp hastalığı bulunmamaktadır.")
else:
    st.write("Sonuç: Kalp hastalığı mevcuttur.")

