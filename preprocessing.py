import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

print("SCRIPT BERJALAN...")

# Pastikan folder data/ ada
if not os.path.exists("data"):
    print("Folder data TIDAK ditemukan — membuat folder.")
    os.makedirs("data")

print("Step 1: Load Dataset")
df = pd.read_excel('data/Dataset_Pengguna_Ojek_Online.xlsx')
print(df.head())

# ---------------------------------------------------------
print("Step 2: Handle Missing Values")
# ---------------------------------------------------------
num_cols = ['Umur', 'Jumlah Pesanan']
cat_cols = ['Pekerjaan', 'Metode Pembayaran', 'Rating Tinggi']

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values selesai")

# ---------------------------------------------------------
print("Step 3: Encoding")
# ---------------------------------------------------------
df_enc = pd.get_dummies(df, columns=['Pekerjaan', 'Metode Pembayaran'])
df_enc['Rating Tinggi'] = df_enc['Rating Tinggi'].map({'Ya':1, 'Tidak':0})
print("Encoding selesai")

# ---------------------------------------------------------
print("Step 4: Outlier Capping")
# ---------------------------------------------------------
def cap_outlier(col):
    Q1 = df_enc[col].quantile(0.25)
    Q3 = df_enc[col].quantile(0.75)
    IQR = Q3 - Q1
    low = Q1 - 1.5 * IQR
    high = Q3 + 1.5 * IQR
    df_enc[col] = np.where(df_enc[col] < low, low, df_enc[col])
    df_enc[col] = np.where(df_enc[col] > high, high, df_enc[col])

cap_outlier('Umur')
cap_outlier('Jumlah Pesanan')
print("Outlier selesai")

# ---------------------------------------------------------
print("Step 5: Standarisasi")
# ---------------------------------------------------------
scaler = StandardScaler()
df_enc[['Umur', 'Jumlah Pesanan']] = scaler.fit_transform(
    df_enc[['Umur', 'Jumlah Pesanan']]
)
print("Standarisasi selesai")

# ---------------------------------------------------------
print("Step 6: Simpan hasil preprocessing")
# ---------------------------------------------------------
df_enc.to_csv('data/preprocessed_data.csv', index=False)
print("File preprocessed_data.csv berhasil dibuat!")

# ---------------------------------------------------------
print("Step 7: Feature Engineering")
# ---------------------------------------------------------
df_fe = df_enc.copy()

df_fe['Rasio_Pesanan_Umur'] = df_fe['Jumlah Pesanan'] / (df_fe['Umur'] + 1)

df_fe.to_csv('data/feature_engineered_data.csv', index=False)
print("File feature_engineered_data.csv berhasil dibuat!")

print("=== SEMUA SELESAI ===")
