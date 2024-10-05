import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
import matplotlib.pyplot as plt  
import seaborn as sns 
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

pio.templates.default = "plotly_white"

st.title('Analisis Penggunaan Sepeda dan Clustering')

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

try:
    data = pd.concat([data_day, data_hour], ignore_index=True)

    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data.dropna(subset=['date'], inplace=True)

    numeric_columns = data.select_dtypes(include=['object']).columns
    for col in numeric_columns:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')  
        except Exception as e:
            st.error(f"Kesalahan saat mengonversi kolom {col}: {e}")

    data_cleaned = data.dropna().drop_duplicates()

    st.write("Informasi DataFrame:")
    st.write(data_cleaned.info())

    if not data_cleaned.select_dtypes(include=['number']).empty:
        corr_matrix = data_cleaned.corr()
        st.write("Korelasi Antar Variabel Numerik:")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Korelasi Antar Variabel Numerik')
        st.pyplot()
    else:
        st.error("Data bersih tidak memiliki kolom numerik untuk menghitung korelasi.")

    st.subheader("Jumlah Pengguna Kasual vs Terdaftar")
    st.write("Total Pengguna Kasual:", data['casual'].sum())
    st.write("Total Pengguna Terdaftar:", data['registered'].sum())

    st.subheader("Perbandingan Penggunaan Sepeda antara Hari Kerja dan Hari Libur")
    total_users_by_working_day = data.groupby('workingday')['cnt'].sum().reset_index()

    fig3 = px.bar(total_users_by_working_day, 
                  x='workingday', 
                  y='cnt', 
                  title='Penggunaan Sepeda berdasarkan Hari Kerja (1 = Ya, 0 = Tidak)',
                  labels={'workingday': 'Hari Kerja', 'cnt': 'Total Pengguna'})
    st.plotly_chart(fig3)

    st.subheader("Clustering Penggunaan Sepeda berdasarkan Faktor Lingkungan dan Musim")

    features = data[['temp', 'hum', 'windspeed', 'season']]

    features = pd.get_dummies(features, columns=['season'], drop_first=True)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42)
    data['cluster'] = kmeans.fit_predict(features_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='temp', y='cnt', hue='cluster', palette='viridis')
    plt.title("Clustering berdasarkan Suhu dan Jumlah Penyewaan")
    st.pyplot(plt.gcf())

    feature_names = features.columns.tolist()

    st.subheader("Prediksi Cluster untuk Input Baru")
    temp = st.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0)
    hum = st.number_input("Humidity (%)", min_value=0, max_value=100, value=30)
    windspeed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0)
    season = st.selectbox("Season", ['spring', 'summer', 'fall', 'winter'])

    new_data = pd.DataFrame({
        'temp': [temp],
        'hum': [hum],
        'windspeed': [windspeed]
    })

    season_dummies = pd.get_dummies([season], prefix='season', drop_first=True)
    season_dummies = season_dummies.reindex(columns=['season_summer', 'season_fall', 'season_winter'], fill_value=0)

    new_data = pd.concat([new_data, season_dummies], axis=1)

    for col in feature_names:
        if col not in new_data.columns:
            new_data[col] = 0

    new_data_scaled = scaler.transform(new_data[feature_names])

    predicted_cluster = kmeans.predict(new_data_scaled)
    st.write(f"Data baru diprediksi berada di cluster: {predicted_cluster[0]}")

    st.write("Lima contoh dari cluster yang sama:")
    st.write(data[data['cluster'] == predicted_cluster[0]].sample(5))

    import joblib
    joblib.dump(kmeans, 'kmeans_model.pkl')
    st.write("Model KMeans telah disimpan.")

    if st.button("Muat KMeans Model"):
        loaded_kmeans = joblib.load('kmeans_model.pkl')
        st.write("Model KMeans berhasil dimuat.")

    st.subheader("Visualisasi Jumlah Penyewaan per Musim")
    rental_per_season = data.groupby('season')['cnt'].sum().reset_index()
    fig_season = px.bar(rental_per_season, 
                        x='season', 
                        y='cnt', 
                        title='Jumlah Penyewaan per Musim',
                        labels={'season': 'Musim', 'cnt': 'Total Penyewaan'})
    st.plotly_chart(fig_season)

    cleaned_file_path = os.path.join("dashboard", "cleaned_data.csv")
    data_cleaned.to_csv(cleaned_file_path, index=False)
    st.write(f"Data bersih disimpan di: {cleaned_file_path}")

except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
