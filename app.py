import streamlit as st
import joblib
import numpy as np

# Load model pipeline lengkap
model = joblib.load("model_pipeline.pkl")  # Pastikan ini pipeline lengkap

st.set_page_config(page_title="Prediksi Risiko Hipertensi", layout="centered")
st.title("🔬 Prediksi Risiko Hipertensi")
st.markdown("Masukkan data kesehatan Anda di bawah untuk mengetahui risiko hipertensi.")

col1, col2 = st.columns(2)

with col1:
    usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=30)
    berat_badan = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=200.0, value=60.0)
    lingkar_pinggang = st.number_input("Lingkar Pinggang (cm)", min_value=30.0, max_value=200.0, value=80.0)
    lingkar_pinggang_ulang = st.number_input("Lingkar Pinggang (Ulang) (cm)", min_value=30.0, max_value=200.0, value=80.0)

with col2:
    tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=40.0, max_value=250.0, value=120.0)
    imt = st.number_input("IMT (kg/m²)", min_value=10.0, max_value=60.0, value=23.0)
    aktivitas_total = st.number_input("Aktivitas Total (menit/hari)", min_value=0.0, max_value=1440.0, value=60.0)

if st.button("🔍 Prediksi Risiko"):
    try:
        input_data = np.array([[usia, berat_badan, lingkar_pinggang, lingkar_pinggang_ulang,
                                tekanan_darah, imt, aktivitas_total]])
        proba = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader("📊 Hasil Prediksi:")
        st.markdown(f"🧪 **Probabilitas Risiko Hipertensi:** `{proba*100:.2f}%`")

        if prediction == 1:
            st.warning("⚠️ **Hasil: Anda berisiko hipertensi.**")
        else:
            st.success("✅ **Hasil: Anda tidak berisiko hipertensi.**")

        st.markdown("---")
        st.caption("Model yang digunakan: LightGBM (dilatih dengan SMOTE & Feature Selection RFE)")

    except Exception as e:
        st.error(f"Terjadi kesalahan pada prediksi: {e}")
