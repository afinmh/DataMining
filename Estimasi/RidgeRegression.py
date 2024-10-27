import numpy as np
import pandas as pd

excel_filename = '/content/harga.xlsx'
df = pd.read_excel(excel_filename)

# Mengisi X dan y
X = df[['Ukuran(m2)', 'Jumlah Kamar', 'Jarak(miles)']].values  # Mengambil kolom luas rumah dan luas halaman
y = df['Harga'].values  # Mengambil kolom harga
y = y.reshape(-1, 1)  # 10 rows, 1 column

X = np.c_[np.ones(X.shape[0]), X]  # Menambahkan kolom
# Parameter Ridge
lambda_reg = 1 
I = np.eye(X.shape[1])  # Matriks identitas
mat_identitas = lambda_reg * I
# Menghitung (X^T X + λI)
XtX = np.dot(X.T, X)    # X^T * X
XtX_lambdaI = XtX + mat_identitas  # X^T * X + λI

# Menghitung X^T * y
Xty = np.dot(X.T, y)  # X^T * y

# Menghitung invers dari (X^T X + λI)
XtX_lambdaI_inv = np.linalg.inv(XtX_lambdaI)

# Menghitung koefisien beta
beta = np.dot(XtX_lambdaI_inv, Xty)

# Menampilkan hasil
print("Koefisien Ridge Regression:")
print("Intercept (β0):", beta[0])
print("Koefisien untuk Ukuran Rumah (β1):", beta[1])
print("Koefisien untuk Kamar (β2):", beta[2])
print("Koefisien untuk Jarak (β3):", beta[3])

b = int(input("Ukuran Rumah (m2) : "))
c = int(input("Jumlah Kamar : "))
d = float(input("Jarak ke Pusat Kota (miles) : "))

a = beta[0] + beta[1] * b + beta[2] * c + beta[3] * d
print("Estimasi Harga : ", a)
