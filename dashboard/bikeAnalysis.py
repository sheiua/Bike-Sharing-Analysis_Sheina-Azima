import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt  
import seaborn as sns 

pio.templates.default = "plotly_white"

import os

print("Current Working Directory: ", os.getcwd())

st.title('Analisis Penggunaan Sepeda')
file_path = os.path.join("C:", "Users", "User", "Downloads", "Bike Sharing Analysis", "day.csv")

# Membaca file CSV
data_day = pd.read_csv(file_path)

data_hour = pd.read_csv("hour.csv")

data = pd.concat([data_day, data_hour], ignore_index=True)


st.write("Lima Baris Pertama dari Data:")
st.write(data.head())
st.write("Dimensi Data (Rows, Columns):", data.shape)
st.write("Informasi Data:")
st.write(data.info())
st.write("Statistik Deskriptif:")
st.write(data.describe())
st.write("Jumlah Missing Values per Kolom:")
st.write(data.isnull().sum())
st.write("Jumlah Nilai Unik per Kolom:")
st.write(data.nunique())

if 'Unnamed' in data.columns:
    data = data.drop(columns=['Unnamed'])

data_cleaned = data.dropna()
data_cleaned = data_cleaned.drop_duplicates()

if 'PM2.5' in data_cleaned.columns:
    data_cleaned = data_cleaned[data_cleaned['PM2.5'] <= 500]

data_cleaned = data_cleaned.dropna(axis=1, how='all')

st.write("Lima Baris Pertama Data Bersih:")
st.write(data_cleaned.head())

st.write("Korelasi Antar Variabel Numerik:")
corr_matrix = data.corr()
st.write(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasi Antar Variabel Numerik')
st.pyplot()

st.write("Data Awal:")
st.write(data.head())

st.subheader("Jumlah Pengguna Kasual vs Terdaftar")
st.write("Total Pengguna Kasual:", data['casual'].sum())
st.write("Total Pengguna Terdaftar:", data['registered'].sum())

st.subheader("Perbandingan Penggunaan Sepeda antara Hari Kerja dan Hari Libur")

total_users_by_working_day = data.groupby('workingday')['cnt'].sum().reset_index()

st.write("Total Pengguna Berdasarkan Hari Kerja dan Hari Libur:")
st.write(total_users_by_working_day)

fig3 = px.bar(total_users_by_working_day, 
                x='workingday', 
                y='cnt', 
                title='Penggunaan Sepeda berdasarkan Hari Kerja (1 = Ya, 0 = Tidak)',
                labels={'workingday': 'Hari Kerja', 'cnt': 'Total Pengguna'})
    
st.plotly_chart(fig3)

st.subheader("Binning pada Jumlah Sepeda yang Digunakan")
bins = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
labels = ['0-50', '51-100', '101-150', '151-200', '201-250', 
          '251-300', '301-350', '351-400', '401-450', '451-500']

data['binned_cnt'] = pd.cut(data['cnt'], bins=bins, labels=labels, right=False)

binned_counts = data['binned_cnt'].value_counts().sort_index()

st.write("Frekuensi Penggunaan Sepeda Berdasarkan Bin:")
st.bar_chart(binned_counts)

binned_counts_df = binned_counts.reset_index()  
binned_counts_df.columns = ['Bin Jumlah Pengguna', 'Frekuensi']  

fig2 = px.bar(binned_counts_df, 
               x='Bin Jumlah Pengguna', 
               y='Frekuensi', 
               title='Frekuensi Penggunaan Sepeda Berdasarkan Bin',
               labels={'Bin Jumlah Pengguna': 'Bin Jumlah Pengguna', 'Frekuensi': 'Frekuensi'})

st.plotly_chart(fig2)
