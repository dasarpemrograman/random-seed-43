# 🧠 Tugas Besar 1 Pembelajaran Mesin: Feed Forward Neural Network

Repositori ini berisi implementasi dan eksperimen model **Feed Forward Neural Network (FFNN)** dari awal (*from scratch*) maupun menggunakan pustaka standar, sebagai bagian dari pemenuhan Tugas Besar 1 mata kuliah Pembelajaran Mesin.

---

## 📑 Daftar Isi
1. [Deskripsi Proyek](#-deskripsi-proyek)
2. [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
3. [Struktur Direktori](#-struktur-direktori)
4. [Dataset](#-dataset)
5. [Cara Setup & Instalasi](#-cara-setup--instalasi)
6. [Cara Menjalankan Program](#-cara-menjalankan-program)
7. [Arsitektur & Modul Kustom](#-arsitektur--modul-kustom)
8. [Hasil & Evaluasi](#-hasil--evaluasi)
9. [Pembagian Tugas Anggota Kelompok](#-pembagian-tugas-anggota-kelompok)

---

## 💡 Deskripsi Proyek
Proyek ini bertujuan untuk membangun, melatih, dan mengevaluasi model Jaringan Saraf Tiruan (*Neural Network*) tipe *Feed Forward*. Kami mengimplementasikan berbagai komponen inti dari arsitektur *deep learning* seperti fungsi aktivasi, inisialisasi bobot, fungsi *loss*, dan regularisasi untuk memahami secara mendalam cara kerja model di balik layar, lalu membandingkannya atau mengintegrasikannya dengan alur kerja penanganan data yang *imbalanced*.

---

## 📦 Teknologi yang Digunakan

| Pustaka / Alat | Kegunaan |
|---|---|
| **Python 3.14+** | Bahasa pemrograman utama |
| **scikit-learn** | *Preprocessing*, metrik evaluasi, & *baseline model* |
| **pandas** | Manipulasi dan analisis data tabular |
| **numpy** | Komputasi numerik dan operasi matriks/tensor |
| **matplotlib / seaborn**| Visualisasi data dan grafik metrik (*loss/accuracy*) |
| **imblearn** | Penanganan data yang tidak seimbang (*imbalanced data*) |
| **marimo / Jupyter** | Lingkungan *notebook* interaktif untuk eksperimen |

---

## 📂 Struktur Direktori

    📁 random-seed-42/
    ├── 📄 README.md                 # Dokumentasi utama proyek
    ├── 📄 pyproject.toml            # Konfigurasi dependensi lingkungan Python
    ├── 📁 data/                     # Direktori untuk menyimpan dataset (di-ignore oleh git)
    ├── 📁 src/                      # Source code utama (Implementasi dari nol)
    │   ├── 📄 model.py              # Kelas utama arsitektur Neural Network
    │   ├── 📄 layers.py             # Implementasi lapisan (Dense/Hidden layers)
    │   ├── 📄 activations.py        # Fungsi aktivasi (Sigmoid, ReLU, Softmax, dll)
    │   ├── 📄 losses.py             # Fungsi kerugian (MSE, Cross-Entropy, dll)
    │   ├── 📄 initializers.py       # Metode inisialisasi bobot (Xavier, He, dll)
    │   └── 📄 regularizers.py       # Teknik regularisasi (L1, L2, Dropout)
    └── 📁 notebooks/                # Eksperimen, preprocessing, dan training
        └── 📄 eksperimen_ffnn.ipynb # Notebook utama untuk menjalankan keseluruhan proses

---

## 📊 Dataset
- **Nama Dataset:** [Masukkan Nama Dataset Anda di sini, misal: *Credit Card Fraud Detection*]
- **Sumber:** [Masukkan Link Kaggle / UCI Machine Learning]
- **Karakteristik:** Dataset ini memiliki fitur sebanyak `[X]` kolom dan `[Y]` baris. Target variabel memiliki masalah *class imbalance* yang kami tangani menggunakan teknik *oversampling/undersampling* melalui modul `imblearn`.

---

## ⚙️ Cara Setup & Instalasi

### Prasyarat
Pastikan sistem Anda telah terinstal **Python 3.14** atau versi yang lebih baru.

### 1. Clone Repositori
    git clone <url-repositori-anda>
    cd random-seed-42

### 2. Buat & Aktifkan Virtual Environment
Sangat disarankan untuk menggunakan *virtual environment* agar dependensi terisolasi.

    python -m venv venv

- **Linux / Mac:**
    source venv/bin/activate

- **Windows:**
    venv\Scripts\activate

### 3. Instal Dependensi
Proyek ini dikelola menggunakan `pyproject.toml`. Jalankan perintah berikut untuk menginstal seluruh pustaka yang dibutuhkan:

    pip install .

---

## ▶️ Cara Menjalankan Program

Seluruh alur pemrosesan data, pelatihan, hingga pengujian model dirangkum dalam bentuk *notebook* interaktif.

1. Buka file *notebook* di dalam folder `notebooks/` menggunakan **Jupyter Notebook**, **JupyterLab**, atau **Marimo**.
2. Pastikan *kernel* yang dipilih adalah *virtual environment* (`venv`) yang baru saja Anda buat.
3. Jalankan seluruh sel sekaligus dengan menekan menu:
   > **Cell → Run All** (atau gunakan pintasan `Shift` + `Enter` secara berurutan pada setiap sel).

---

## 🏗️ Arsitektur & Modul Kustom
Sebagai bagian dari *learning outcome*, kami menulis implementasi spesifik untuk komponen Neural Network, meliputi:
- **`model.py` & `layers.py`**: Mengelola *forward propagation* dan *backward propagation*.
- **`activations.py` & `losses.py`**: Berisi kalkulasi matematis untuk fungsi aktivasi beserta turunannya, dan perhitungan *error*.
- **`initializers.py` & `regularizers.py`**: Mengontrol pembentukan bobot awal untuk mencegah *vanishing/exploding gradient* serta fungsi penalti untuk mencegah *overfitting*.

---

## 📈 Hasil & Evaluasi
*(Bagian ini dapat Anda lengkapi setelah eksperimen selesai)*

- **Metrik Utama:** *Accuracy, Precision, Recall, F1-Score*.
- **Kesimpulan Singkat:** Model berhasil dioptimasi dengan akurasi mencapai `[XX]%`. Penggunaan `imblearn` (misal: SMOTE) terbukti meningkatkan metrik *Recall* pada kelas minoritas dari `[X]%` menjadi `[Y]%`.

---

## 👥 Pembagian Tugas Anggota Kelompok

| Nama / NIM | Rincian Pekerjaan |
|---|---|
| **Wisa Ahmaduta Dinutama**<br>18223003 | • Melakukan dataset *preprocessing*<br>• Melakukan *experiments*<br>• Melakukan pengujian model<br>• Mengerjakan laporan |
| **Muhammad Faiz Alfikrona**<br>18223009 | • Mengerjakan `regularizers.py`, `layers.py`, dan `model.py`<br>• Melakukan pengujian model<br>• Mengerjakan laporan |
| **Ni Made Sekar Jelita**<br>18223101 | • Mengerjakan `activations.py`, `initializers.py`, dan `losses.py`<br>• Mengerjakan laporan |

---
*Dibuat untuk memenuhi Tugas Besar 1 Pembelajaran Mesin.*
