# Kalori Terbakar Prediction

Proyek ini bertujuan untuk memprediksi kalori yang terbakar berdasarkan sejumlah fitur menggunakan teknik machine learning. Algoritma utama yang digunakan adalah Support Vector Regression (SVR), serta teknik Principal Component Analysis (PCA) untuk pengurangan dimensi dan GridSearchCV untuk optimasi parameter.

## Deskripsi Proyek

Dalam proyek ini, kami menggunakan dataset yang berisi informasi tentang aktivitas fisik, seperti durasi, intensitas, dan variabel lainnya, untuk memprediksi jumlah kalori yang terbakar. Dengan menggunakan model SVR, PCA, dan GridSearchCV, diharapkan hasil prediksi lebih akurat dan model dapat bekerja lebih efisien dengan mengurangi dimensi dataset.

## Alur Proyek

1. **Preprocessing Data:** Memuat dan membersihkan dataset, termasuk menangani nilai yang hilang dan normalisasi fitur.
2. **Pengurangan Dimensi (PCA):** Menggunakan PCA untuk mengurangi dimensi fitur, menjaga informasi penting, dan mengurangi kompleksitas komputasi.
3. **Modeling (SVR):** Melatih model SVR untuk memprediksi kalori yang terbakar.
4. **Optimasi (GridSearchCV):** Menggunakan GridSearchCV untuk menemukan parameter terbaik bagi model SVR, meningkatkan akurasi prediksi.

## Instalasi

1. Clone repositori ini:
    ```bash
    git clone https://github.com/username/kalori-terbakar-prediksi.git
    cd kalori-terbakar-prediksi
    ```

2. Instal dependensi:
    ```bash
    pip install -r requirements.txt
    ```

## Penggunaan

1. **Dataset:** Letakkan file dataset dalam folder `data`. Pastikan dataset memiliki kolom yang sesuai untuk fitur yang digunakan.

2. **Jalankan Script Utama:** Gunakan script berikut untuk menjalankan pipeline prediksi:
    ```bash
    python main.py
    ```

3. **Parameter GridSearch:** Anda dapat menyesuaikan parameter pencarian GridSearch di file `config.py`, seperti `C`, `epsilon`, atau `kernel`.

## Struktur Folder

- `data/`: Folder untuk dataset.
- `scripts/`: Folder berisi script utama dan pendukung.
- `models/`: Folder untuk menyimpan model yang telah dilatih.
- `config.py`: File konfigurasi untuk parameter PCA dan GridSearchCV.

## Hasil Evaluasi

Hasil evaluasi model akan disimpan di folder `results/`, termasuk metrik evaluasi seperti Mean Squared Error (MSE) dan Mean Absolute Error (MAE).

## Dependensi

Proyek ini memerlukan pustaka berikut:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## Penulis

- **Nama Anda** - [GitHub](https://github.com/username)

## Lisensi

Proyek ini dilisensikan di bawah MIT License. Silakan lihat file `LICENSE` untuk informasi lebih lanjut.
