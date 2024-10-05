import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
import streamlit as st
import matplotlib.pyplot as plt  
import seaborn as sns 
import os

pio.templates.default = "plotly_white"

st.title('Analisis dan Prediksi Penggunaan Sepeda')

# Menampilkan direktori kerja saat ini
st.write("Current Working Directory: ", os.getcwd())

# Jalur ke file CSV
file_path_day = os.path.join("dashboard", "day.csv")
file_path_hour = os.path.join("dashboard", "hour.csv")

# Membaca file day.csv
try:
    data_day = pd.read_csv(file_path_day)
    st.write("Data hari berhasil dimuat.")
except FileNotFoundError:
    st.error(f"File {file_path_day} tidak ditemukan. Pastikan jalur file sudah benar.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("File tidak memiliki data. Periksa file CSV.")
    st.stop()
except pd.errors.ParserError:
    st.error("Kesalahan saat mem-parsing file CSV. Pastikan format file benar.")
    st.stop()

# Membaca file hour.csv
try:
    data_hour = pd.read_csv(file_path_hour)
    st.write("Data jam berhasil dimuat.")
except FileNotFoundError:
    st.error(f"File {file_path_hour} tidak ditemukan. Pastikan jalur file sudah benar.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("File tidak memiliki data. Periksa file CSV.")
    st.stop()
except pd.errors.ParserError:
    st.error("Kesalahan saat mem-parsing file CSV. Pastikan format file benar.")
    st.stop()

# Menggabungkan data
try:
    data = pd.concat([data_day, data_hour], ignore_index=True)

    # Membersihkan data
    if 'Unnamed' in data.columns:
        data = data.drop(columns=['Unnamed'])

    data_cleaned = data.dropna()
    data_cleaned = data_cleaned.drop_duplicates()

    # Tampilkan informasi dan analisis data
    st.write("Lima Baris Pertama dari Data:")
    st.write(data_cleaned.head())
    
    # Cek kolom numerik dan korelasi
    if not data_cleaned.select_dtypes(include=['number']).empty:
        corr_matrix = data_cleaned.corr()
        st.write("Korelasi Antar Variabel Numerik:")
        st.write(corr_matrix)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Korelasi Antar Variabel Numerik')
        st.pyplot()
    else:
        st.error("Data bersih tidak memiliki kolom numerik untuk menghitung korelasi.")

    st.write("Data Awal:")
    st.write(data.head())

    # Hitung rata-rata untuk memberikan prediksi sederhana
    average_rental_count = data_cleaned['rental_count'].mean()
    st.write(f"Rata-rata jumlah sewa sepeda: {average_rental_count:.2f}")

    # Streamlit app untuk prediksi sederhana
    st.subheader("Prediksi Jumlah Sewa Sepeda (Sederhana)")

    # User input untuk fitur
    temperature = st.number_input("Temperature (°C)", min_value=-30.0, max_value=50.0, value=20.0)
    humidity = st.number_input("Humidity (%)", min_value=0, max_value=100, value=30)
    wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)
    season = st.selectbox("Season", ['spring', 'summer', 'fall', 'winter'])

    # Prediction button
    if st.button("Predict"):
        # Menggunakan rata-rata sebagai prediksi sederhana
        st.write(f"Prediksi jumlah sewa sepeda: {average_rental_count:.2f}")

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
