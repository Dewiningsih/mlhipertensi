import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load("model_final.pkl")

# Title
st.set_page_config(page_title="Prediksi Hipertensi", layout="centered")
st.title("🩺 Prediksi Risiko Hipertensi")
st.write("Masukkan data Anda untuk melihat risiko hipertensi berdasarkan model machine learning.")

# Input user
col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia (tahun)", min_value=10, max_value=120, value=30)
    berat = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0, value=60.0)
    lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=30.0, max_value=150.0, value=85.0)
    lingkar_pinggang_ulang = st.number_input("Lingkar Pinggang (Ulang) (cm)", min_value=30.0, max_value=150.0, value=85.0)

with col2:
    tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=60.0, max_value=250.0, value=120.0)
    imt = st.number_input("IMT (kg/m²)", min_value=10.0, max_value=60.0, value=22.0)
    aktivitas_total = st.number_input("Aktivitas Total (menit/hari)", min_value=0.0, max_value=1440.0, value=30.0)

# Prediksi
if st.button("🔍 Prediksi"):
    input_data = np.array([[usia, berat, lingkar_pinggang, lingkar_pinggang_ulang,
                            tekanan_darah, imt, aktivitas_total]])

    proba = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.subheader("📈 Hasil Prediksi")
    st.write(f"**Probabilitas Risiko Hipertensi:** `{proba*100:.2f}%`")

    if pred == 1:
        st.warning("⚠️ Anda berisiko mengalami hipertensi.")
    else:
        st.success("✅ Anda tidak menunjukkan risiko hipertensi.")

    st.caption("Model: LGBM dengan SMOTE & RFE (7 fitur utama)")
