import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import io
from PIL import Image

def pca(ch, n_komp):
    mean_col = np.mean(ch, axis=0)
    ch_center = ch - mean_col

    n = ch_center.shape[0]
    kov = (ch_center.T @ ch_center) / (n - 1)

    def power_iterasi(A, iterasi=50):
        b = np.random.rand(A.shape[1])
        for _ in range(iterasi):
            b = A @ b
            b = b / np.linalg.norm(b)
        return b

    eigvec_list = []
    eigval_list = []

    A = kov.copy()
    for _ in range(n_komp):
        v = power_iterasi(A)
        lambda_val = v @ A @ v
        eigvec_list.append(v)
        eigval_list.append(lambda_val)
        A = A - lambda_val * np.outer(v, v)

    eigvecs = np.stack(eigvec_list, axis=1)
    eigvals = np.array(eigval_list)

    proyeksi = ch_center @ eigvecs
    rekonstruksi = proyeksi @ eigvecs.T + mean_col

    total_var = np.trace(kov)
    var_ambil = np.sum(eigvals) / total_var

    return np.clip(rekonstruksi, 0, 255), var_ambil, eigvals

@st.cache_data(show_spinner=False)
def image_to_bytes(img_array):
    buffer = io.BytesIO()
    Image.fromarray(img_array).save(buffer, format="PNG")
    return buffer.getvalue()

def start_app():
    st.set_page_config(page_title="Kompresi Gambar dengan PCA", layout="centered")
    st.markdown("""
        <style>
        .stImage > img {
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .info-box {
            background-color: rgba(255,255,255,0.07);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(150,150,150,0.3);
            margin: 1rem 0;
            backdrop-filter: blur(5px);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Kompresi Gambar dengan PCA")

    st.sidebar.header("Pengaturan Input")
    file_input = st.sidebar.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if file_input is not None:
        file_arr = np.asarray(bytearray(file_input.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        tinggi, lebar, _ = original_img.shape

        st.subheader("Gambar Asli")
        st.image(original_img, use_container_width=True)
        st.markdown(f"<div class='info-box'>Dimensi Gambar Asli: <b>{original_img.shape}</b></div>", unsafe_allow_html=True)

        jumlah_komponen = st.sidebar.slider("Jumlah Komponen PCA", 1, min(tinggi, lebar), 50)

        if st.sidebar.button("Mulai Kompresi"):
            with st.spinner("Sedang mengompresi gambar..."):
                mulai = time.time()

                hasil_kompresi = np.zeros_like(original_img, dtype=np.uint8)
                semua_var = []
                semua_eigen_merah = None

                for idx_channel in range(3):
                    data_channel = original_img[:, :, idx_channel].astype(np.float64)
                    hasil_rekonstruksi, persentase_var, eigs = pca(data_channel, jumlah_komponen)
                    hasil_kompresi[:, :, idx_channel] = hasil_rekonstruksi.astype(np.uint8)
                    semua_var.append(persentase_var)
                    if idx_channel == 0:
                        semua_eigen_merah = eigs  # simpan eigen untuk channel merah

                durasi = time.time() - mulai
                rata_rata_var = np.mean(semua_var) * 100

                st.subheader("Gambar Hasil Kompresi")
                st.image(hasil_kompresi, caption=f"{jumlah_komponen} Komponen", use_container_width=True)
                st.markdown(f"<div class='info-box'>Dimensi Rekonstruksi: <b>{hasil_kompresi.shape}</b><br>Waktu Kompresi: <b>{durasi:.2f} detik</b></div>", unsafe_allow_html=True)

                st.subheader("Informasi Variansi per Channel")
                fig_var, ax_var = plt.subplots()
                ax_var.bar(['R', 'G', 'B'], [v * 100 for v in semua_var], color=['#e74c3c', '#27ae60', '#2980b9'])
                ax_var.set_ylabel("Persentase Variansi (%)")
                ax_var.set_ylim(0, 100)
                st.pyplot(fig_var)

                # Gunakan hasil eigval dari channel R (merah) yang sudah dihitung
                kumulatif_var = np.cumsum(semua_eigen_merah) / np.sum(semua_eigen_merah)
                fig_kumulatif, ax_kumulatif = plt.subplots()
                ax_kumulatif.plot(kumulatif_var, color='#e74c3c')
                ax_kumulatif.set_title("Cumulative Explained Variance (Channel R)")
                ax_kumulatif.set_xlabel("Jumlah Komponen")
                ax_kumulatif.set_ylabel("Kumulatif Variansi")
                ax_kumulatif.grid(True)
                st.pyplot(fig_kumulatif)

                ukuran_asli = tinggi * lebar * 3
                ukuran_kompres = (tinggi * jumlah_komponen + jumlah_komponen * lebar) * 3
                persen_reduksi = 100 - (ukuran_kompres / ukuran_asli * 100)

                kol1, kol2 = st.columns(2)
                with kol1:
                    st.markdown(f"<div class='info-box'>Informasi Rata-rata Dipertahankan: <b>{rata_rata_var:.2f}%</b></div>", unsafe_allow_html=True)
                with kol2:
                    st.markdown(f"<div class='info-box'>Estimasi Reduksi Ukuran Data: <b>{persen_reduksi:.2f}%</b></div>", unsafe_allow_html=True)

                byte_kompres = image_to_bytes(hasil_kompresi)
                st.download_button(
                    label="Unduh Gambar Kompresi",
                    data=byte_kompres,
                    file_name="compressed_image.png",
                    mime="image/png"
                )
    else:
        st.info("Silakan upload gambar melalui sidebar.")
