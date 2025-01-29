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

## Struktur Folder

- `colabs_file/`: Folder untuk menyimpan file terkait Colab atau notebook eksperimen.
- `model/`: Folder untuk menyimpan model dan scaler, termasuk:
  - `pca_best.pkl`: Model PCA yang sudah dioptimalkan.
  - `scaler.pkl`: Scaler yang digunakan untuk preprocessing.
  - `svr_model.pkl`: Model SVR yang telah dilatih.
- `static/`: Folder untuk aset statis, termasuk:
  - `css/`: File CSS untuk styling.
  - `img/`: Folder untuk gambar.
  - `js/`: Script JavaScript.
  - `lib/`: Library tambahan.
  - `scss/`: File SCSS untuk styling tambahan.
- `templates/`: Folder untuk file template HTML, termasuk:
  - `index.html`: Halaman utama aplikasi.
  - `modal.html`: File modal HTML untuk dialog atau popup.
- `venv/`: Lingkungan virtual Python (tidak termasuk dalam versi kontrol).

## Penggunaan

1. **Dataset:** Letakkan file dataset dalam folder yang sesuai di `colabs_file` atau di lokasi lain yang diinginkan.

2. **Jalankan Script Utama:** Gunakan script berikut untuk menjalankan pipeline prediksi:
    ```bash
    python main.py
    ```

3. **Web Interface:** File HTML dan aset statis tersedia di folder `templates` dan `static` untuk membuat antarmuka prediksi.

## Dependensi

Proyek ini memerlukan pustaka berikut:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `flask` (jika menggunakan web interface)

## Penulis

- **Sandhya Pratama Hutagalung** - [GitHub](https://github.com/Sandhya-Pratama)

