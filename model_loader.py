# -----------------------------------------------------------------
# NAMA FILE: model_loader.py
# -----------------------------------------------------------------
import streamlit as st
import gdown
import os
import shutil

# --- ID Google Drive Anda ---
SAHAM_DRIVE_ID = "1tPO-kcpdgSpsJ9k0wvh8fef-AeeUoH4x"
# (Kita tidak perlu ID Teks untuk tugas ini)

# --- Path Lokal di Server Streamlit ---
SAHAM_LOCAL_PATH = "models/saham_weights"

@st.cache_resource(show_spinner="Mengunduh Model Prediksi Saham (hanya sekali)...")
def download_stock_models():
    """
    Mengecek jika folder model saham ada. Jika tidak, unduh dari GDrive.
    """
    
    if os.path.exists(SAHAM_LOCAL_PATH):
        print("Folder 'saham_weights' sudah ada. Melewatkan proses unduh.")
        return True

    print(f"Folder '{SAHAM_LOCAL_PATH}' tidak ditemukan. Mengunduh...")
    
    # Buat folder 'models' sebagai induk jika belum ada
    os.makedirs("models", exist_ok=True)
    
    try:
        # Unduh Model Saham
        gdown.download_folder(id=SAHAM_DRIVE_ID, output=SAHAM_LOCAL_PATH, quiet=False, use_cookies=False)
        
        if os.path.exists(SAHAM_LOCAL_PATH):
            print("Berhasil mengunduh model saham.")
            return True
        else:
            st.error(f"Download GDrive selesai, TAPI folder '{SAHAM_LOCAL_PATH}' tidak ditemukan.")
            st.error("Pastikan nama folder di Google Drive Anda adalah 'saham_weights'.")
            return False
        
    except Exception as e:
        st.error(f"Terjadi error saat mengunduh model: {e}")
        st.error("Pastikan link Google Drive 'saham_weights' Anda disetel ke 'Siapa saja yang memiliki link'.")
        return False