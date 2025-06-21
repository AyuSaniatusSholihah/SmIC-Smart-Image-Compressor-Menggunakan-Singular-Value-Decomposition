# 📦 PROJECT BASED LEARNING 2  
## **SmIC – Smart Image Compressor Menggunakan Singular Value Decomposition**

Aplikasi ini merupakan sistem kompresi gambar digital berbasis **Singular Value Decomposition (SVD)** yang dirancang untuk mengecilkan ukuran file tanpa mengurangi kualitas visual secara signifikan. Dikembangkan dalam rangka **Project Based Learning 2**, proyek ini menyajikan antarmuka interaktif untuk eksperimen langsung terhadap proses kompresi gambar.

---

## 🚀 Teknologi yang Digunakan
- **Python**  
- **NumPy** – komputasi numerik  
- **OpenCV** – pemrosesan citra  
- **Pillow (PIL)** – manipulasi gambar  
- **Matplotlib** – visualisasi data  
- **Streamlit** – antarmuka aplikasi  
- **HTML, CSS, JavaScript** – peningkatan tampilan antarmuka

---

## 🗂️ Struktur File

| File         | Deskripsi                                               |
|--------------|---------------------------------------------------------|
| `main.py`    | File utama untuk menjalankan aplikasi Streamlit         |
| `app.py`     | Fungsi pemrosesan gambar dan SVD                        |
| `program.py` | Modul tambahan dan logika pendukung                     |

---

## 📸 Fitur Utama
- Unggah gambar digital (JPG/PNG)
- Atur jumlah komponen kompresi (nilai \( k \))
- Tampilkan hasil kompresi berdampingan dengan gambar asli
- Statistik kompresi: waktu proses, variansi dipertahankan, pengurangan ukuran file

---

## 👨‍🏫 Dosen Pengampu  
**Drs. Bambang Harjito, M.App.Sc, PhD**

---

## 👥 Disusun Oleh – Kelompok 3
- **AYU SANIATUS SHOLIHAH** — L0124005  
- **FADHIL RUSADI** — L0124013  
- **MUHAMAD NABIL FANNANI** — L0124135

---

## ▶️ Cara Menjalankan Aplikasi

```bash
python app.py
