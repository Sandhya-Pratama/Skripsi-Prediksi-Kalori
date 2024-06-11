import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import csv

with open('../calories+exercise.csvv', mode='r') as file:
    combined_data = csv.reader(file)
    for row in reader:
        print(row)

combined_data.head()

#cek data
combined_data.shape

# Menampilkan informasi data
print("Informasi Data:")
print(combined_data.info())

combined_data.replace({'Gender':{'male':0,'female':1}},inplace=True)
combined_data.drop(['User_ID'], axis=1, inplace=True)
combined_data.head()

# Menampilkan statistik deskriptif
print("\nStatistik Deskriptif:")
print(combined_data.describe().T
)

X=combined_data.drop(['Calories'],axis=1)
Y=combined_data['Calories']

print(X)

print(Y)

# Random sampling
sample_size = int(0.9 * len(combined_data))  # Contoh: 80% dari dataset asli
X_sampled, _, y_sampled, _ = train_test_split(X, Y, train_size=sample_size, random_state=0)

# 4. Preprocessing Data (Scaling)
scaler = StandardScaler()
X_sampled_scaled = scaler.fit_transform(X_sampled)

# Split data menjadi training dan test set
X_train, X_test, y_train, y_test = train_test_split(X_sampled_scaled, y_sampled, test_size=0.2, random_state=0)

X_train.shape

X_test.shape

# Bangun model SVR dengan kernel RBF
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

# Evaluasi model
y_pred = svr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (Without StandardScaler):", mse)
print("R2 Score (Without StandardScaler):", r2)

"""**PCA**"""

pca = PCA()

# Set up parameter grid
param_grid = {
    'n_components': np.arange(2, X_train.shape[1] + 1)  # Mulai dari 2 hingga jumlah fitur
}

# Membuat objek GridSearchCV
grid_search = GridSearchCV(
    pca,
    param_grid,
    cv=10,
    verbose=1)

grid_search.fit(X_train)

best_n_components = grid_search.best_params_['n_components']
print("Best n_components:", best_n_components)

# PCA
pca = PCA(n_components=best_n_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# pca_best = PCA(n_components=best_n_components)
# X_train_pca = pca_best.fit_transform(X_train_scaled)
# X_test_pca = pca_best.transform(X_test_scaled)

# Evaluasi model SVR dengan fitur PCA
svr_pca = SVR(kernel='rbf')

svr_pca.fit(X_train_pca, y_train)
mae_svr_pca = mean_absolute_error(y_test, svr_pca.predict(X_test_pca))
mse_svr_pca = mean_squared_error(y_test, svr_pca.predict(X_test_pca))
r2_svr_pca = r2_score(y_test, svr_pca.predict(X_test_pca))
print("\nEvaluation with SVR + PCA:")
print("Mean Absolute Error (MAE) on test data:", mae_svr_pca)
print("Mean Squared Error (MSE) on test data:", mse_svr_pca)
print("R-squared (R2) score on test data:", r2_svr_pca)

"""Menggunakan hyperparameter"""

# Buat dictionary parameter untuk GridSearchCV untuk SVR
param_grid_svr = {
                  'C': [0.1, 1, 10],
                  'epsilon': [0.01, 0.1, 1],
                  'gamma': [0.1, 1, 10],
                  'kernel': ['rbf']}

# Buat objek k-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Buat objek GridSearchCV untuk mencari hyperparameter terbaik untuk SVR dengan k-fold cross-validation
grid_svr = GridSearchCV(estimator=svr, param_grid=param_grid_svr,refit=True, verbose=2, cv=kf)

grid_svr.fit(X_train_pca, y_train)

best_svr = grid_svr.best_estimator_
print("Best SVR parameters:", grid_svr.best_params_)

#  menggunakan model terbaik
y_pred = best_svr.predict(X_test_pca)

# Hitung dan cetak metrik evaluasi
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

import pickle

# Menyimpan model ke dalam file pickle
with open('pca+svr+gridsearchcv.pkl', 'wb') as file:
    pickle.dump(grid_svr, file)

joblib.dump(grid_svr, 'grid_svr_model.pkl')