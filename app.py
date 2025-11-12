# -----------------------------------------------------------------
# NAMA FILE: app.py
# -----------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from model_loader import download_stock_models # Import helper download

# Import layer-layer Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, GRU, Bidirectional, Input, 
    GroupNormalization
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# --- Bagian 1: Pengaturan Awal & Fungsi Pemuatan Data ---

st.set_page_config(page_title="Prediksi Saham BBCA", layout="wide")
st.title("Aplikasi Prediksi Saham BBCA")

# Path ke file data dan model
PATH_DATA_TRAIN = 'Data Historis BBCA_Train2.csv'
PATH_DATA_TEST = 'Data Historis BBCA_Test2.csv'
PATH_WEIGHTS = 'models/saham_weights/'

# Fungsi untuk memuat dan melatih scaler (menggunakan cache Streamlit)
@st.cache_resource
def get_trained_scaler():
    """
    Memuat data train HANYA untuk melatih (fit) scaler.
    Scaler ini disimpan di cache agar bisa digunakan untuk un-scaling nanti.
    """
    try:
        train_data = pd.read_csv(PATH_DATA_TRAIN, sep=',')
        train_data.columns = train_data.columns.str.strip()
        train_data = train_data.loc[:, ~train_data.columns.str.contains('^Unnamed', na=False)]
        
        # Hapus kolom non-numerik
        data_to_fit = train_data.drop(['Adj Close', 'Date'], axis=1)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_to_fit)
        print("Scaler berhasil dilatih.")
        return scaler
    except FileNotFoundError:
        st.error(f"File data train '{PATH_DATA_TRAIN}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat atau melatih scaler: {e}")
        return None

# Fungsi untuk memuat dan memproses data train/test
@st.cache_data
def load_and_process_data(file_path, scaler, is_train=True):
    """Memuat dan memproses data train atau test."""
    try:
        data = pd.read_csv(file_path, sep=',')
        data.columns = data.columns.str.strip()
        data = data.loc[:, ~data.columns.str.contains('^Unnamed', na=False)]
        
        data_processed = data.drop(['Adj Close', 'Date'], axis=1)
        
        if is_train:
            data_scaled = scaler.transform(data_processed)
        else:
            # Untuk data tes, kita juga perlu .transform
            data_scaled = scaler.transform(data_processed)
        
        return data_scaled
    except FileNotFoundError:
        st.error(f"File data '{file_path}' tidak ditemukan.")
        return None
    except Exception as e:
        st.error(f"Gagal memproses data: {e}")
        return None

def construct_time_frames(data, frame_size=64):
    """Membagi data menjadi sequence X dan y."""
    x_data, y_data = [], []
    for i in range(frame_size, len(data)):
        x_data.append(data[i-frame_size:i])
        y_data.append(data[i, 0]) # Target adalah harga 'Open' (kolom 0)
    return np.array(x_data), np.array(y_data)

# --- Bagian 2: Definisi Arsitektur Model & Pemuatan ---

input_shape = (64, 5) # (64 hari, 5 fitur)

# Definisi arsitektur (harus sama dengan saat training)
layers_lstm = [
    LSTM(units=64, return_sequences=True), GroupNormalization(), Dropout(0.2),
    LSTM(units=64, return_sequences=True), GroupNormalization(), Dropout(0.2),
    LSTM(units=64), GroupNormalization(), Dropout(0.2), Dense(units=1)
]
layers_gru = [
    GRU(units=64, return_sequences=True), GroupNormalization(), Dropout(0.2),
    GRU(units=64, return_sequences=True), GroupNormalization(), Dropout(0.2),
    GRU(units=64), GroupNormalization(), Dropout(0.2), Dense(units=1)
]
layers_bidirectional = [
    Bidirectional(LSTM(units=64, return_sequences=True)), GroupNormalization(), Dropout(0.2),
    Bidirectional(LSTM(units=64, return_sequences=True)), GroupNormalization(), Dropout(0.2),
    Bidirectional(LSTM(units=64)), GroupNormalization(), Dropout(0.2), Dense(units=1)
]

# Dictionary untuk mempermudah
MODEL_CONFIGS = {
    "LSTM": {"name": "lstm_model", "layers": layers_lstm},
    "GRU": {"name": "gru_model", "layers": layers_gru},
    "Bidirectional-LSTM": {"name": "bidirectional_model", "layers": layers_bidirectional}
}

@st.cache_resource
def build_and_load_model(model_name, layers):
    """Membangun arsitektur dan memuat bobot H5."""
    model = Sequential([Input(shape=input_shape)] + layers)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber')
    
    weights_path = os.path.join(PATH_WEIGHTS, f"{model_name}_callback.h5")
    if not os.path.exists(weights_path):
        st.error(f"File bobot tidak ditemukan: {weights_path}.")
        return None
        
    model.load_weights(weights_path)
    print(f"Model {model_name} berhasil dimuat.")
    return model

# --- Bagian 3: Fungsi Akurasi dan Prediksi ---

@st.cache_data
def calculate_accuracy(_model, x_test, y_test, scaler, n_features=5):
    """Menghitung 'Akurasi' (RMSE) pada data tes dan mengubahnya ke Rupiah."""
    y_pred_scaled = _model.predict(x_test)
    
    # Unscale y_test (harga asli)
    dummy_test = np.zeros((len(y_test), n_features))
    dummy_test[:, 0] = y_test.flatten()
    y_test_rupiah = scaler.inverse_transform(dummy_test)[:, 0]
    
    # Unscale y_pred (harga prediksi)
    dummy_pred = np.zeros((len(y_pred_scaled), n_features))
    dummy_pred[:, 0] = y_pred_scaled.flatten()
    y_pred_rupiah = scaler.inverse_transform(dummy_pred)[:, 0]
    
    # Hitung RMSE dalam Rupiah
    rmse_rupiah = np.sqrt(mean_squared_error(y_test_rupiah, y_pred_rupiah))
    return rmse_rupiah, y_test_rupiah, y_pred_rupiah

def predict_future(model,
                   initial_sequence,
                   days_to_predict,
                   scaler,
                   frame_size=64,
                   n_features=5):
    """Melakukan prediksi berulang untuk N hari ke depan dan meng-unscale."""
    prediksi_scaled_list = []
    current_sequence = initial_sequence.copy()
    
    for _ in range(days_to_predict):
        pred_input = np.reshape(current_sequence, (1, frame_size, n_features))
        pred_harga_scaled = model.predict(pred_input, verbose=0)
        
        prediksi_scaled_list.append(pred_harga_scaled[0, 0])
        
        # Buat baris fitur baru
        fitur_baru = np.full((1, n_features), pred_harga_scaled[0, 0])
        
        # Tambahkan ke sequence
        current_sequence = np.append(current_sequence, fitur_baru, axis=0)
        current_sequence = current_sequence[1:] # Hapus hari pertama
        
    # Unscale hasil prediksi
    pred_array = np.array(prediksi_scaled_list).reshape(-1, 1)
    dummy_array = np.zeros((len(pred_array), n_features))
    dummy_array[:, 0] = pred_array[:, 0]
    prediksi_rupiah = scaler.inverse_transform(dummy_array)[:, 0]
    
    return prediksi_rupiah

# --- Bagian 4: Tampilan Utama Streamlit ---

# Unduh model terlebih dahulu
models_ready = download_stock_models()

if models_ready:
    # 1. Muat scaler (hanya sekali)
    scaler = get_trained_scaler()
    
    if scaler:
        # 2. Muat data train dan test (hanya sekali)
        train_data = load_and_process_data(PATH_DATA_TRAIN, scaler, is_train=True)
        test_data = load_and_process_data(PATH_DATA_TEST, scaler, is_train=False)
        
        if train_data is not None and test_data is not None:
            # 3. Buat sequence data tes untuk evaluasi
            x_test, y_test = construct_time_frames(test_data)
            
            # --- Sidebar (Input Pengguna) ---
            st.sidebar.header("Pengaturan Prediksi")
            
            model_choice = st.sidebar.selectbox(
                "1. Pilih model untuk prediksi:",
                ("Bidirectional-LSTM", "LSTM", "GRU")
            )
            
            days_to_predict = st.sidebar.slider(
                "2. Pilih jumlah hari prediksi:",
                min_value=1, max_value=30, value=7
            )
            
            run_button = st.sidebar.button("Jalankan Prediksi")
            
            # --- Halaman Utama (Output) ---
            
            # 4. Muat model yang dipilih
            config = MODEL_CONFIGS[model_choice]
            model = build_and_load_model(config["name"], config["layers"])
            
            if model:
                st.subheader(f"Model Aktif: {model_choice}")
                
                # 5. Hitung dan tampilkan "Akurasi" (RMSE)
                with st.spinner("Menghitung akurasi model pada data tes..."):
                    rmse_rp, y_test_rp, y_pred_rp = calculate_accuracy(
                        model, x_test, y_test, scaler, n_features=input_shape[1]
                    )
                
                st.metric(
                    label=f"Akurasi Model (RMSE pada data tes historis)",
                    value=f"Rp {rmse_rp:,.2f}",
                    help="Ini adalah rata-rata selisih harga (error) prediksi model saat diuji pada data historis."
                )

                # 6. Tampilkan plot Akurasi (Prediksi vs Asli di data tes)
                with st.expander("Lihat Plot Akurasi pada Data Tes"):
                    fig_test, ax_test = plt.subplots(figsize=(12, 6))
                    ax_test.plot(y_test_rp, color='red', label='Harga Asli (Tes)')
                    ax_test.plot(y_pred_rp, color='blue', label=f'Prediksi {model_choice}')
                    ax_test.set_title('Perbandingan Prediksi vs Harga Asli (Data Tes)')
                    ax_test.set_ylabel('Harga Saham (Rp)')
                    ax_test.legend()
                    st.pyplot(fig_test)

                # 7. Jalankan prediksi masa depan jika tombol ditekan
                if run_button:
                    st.markdown("---")
                    st.subheader(f"Hasil Prediksi {days_to_predict} Hari ke Depan")
                    
                    with st.spinner(f"Memprediksi {days_to_predict} hari ke depan..."):
                        # Ambil data 64 hari terakhir dari data train sebagai titik awal
                        input_seq = train_data[-64:]
                        
                        prediksi_rupiah = predict_future(
                            model, input_seq, days_to_predict, scaler, n_features=input_shape[1]
                        )
                    
                    st.success("Prediksi selesai!")
                    
                    # Tampilkan plot prediksi masa depan
                    fig_future, ax_future = plt.subplots(figsize=(12, 6))
                    ax_future.plot(range(1, days_to_predict + 1), prediksi_rupiah, marker='o', label=f'Prediksi {model_choice} (Rupiah)')
                    ax_future.set_title(f'Prediksi {days_to_predict} Hari ke Depan (Harga \'Open\' BBCA)')
                    ax_future.set_xlabel('Hari ke-')
                    ax_future.set_ylabel('Harga Saham (Rp)')
                    ax_future.legend()
                    ax_future.grid(True)
                    st.pyplot(fig_future)
                    
                    # Tampilkan data tabel
                    st.subheader("Data Prediksi (Rupiah)")
                    df_prediksi = pd.DataFrame({
                        "Hari ke-": range(1, days_to_predict + 1),
                        "Prediksi Harga (Rp)": [f"Rp {harga:,.0f}" for harga in prediksi_rupiah]
                    })
                    st.dataframe(df_prediksi, width=400)
            
            else:
                st.error("Gagal memuat model. Periksa file .h5 di Google Drive.")
        else:
            st.error("Gagal memuat data. Pastikan file CSV ada di repositori GitHub.")
    else:
        st.error("Gagal memuat scaler. Pastikan file 'Data Historis BBCA_Train2.csv' ada.")
else:
    st.error("Gagal mengunduh file model dari Google Drive. Cek log.")