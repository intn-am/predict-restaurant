import streamlit as st
import joblib

# =======================
# Load Model & Encoder
# =======================
model = joblib.load("best_random_forest_model.pkl")
le_menu = joblib.load("menu_category_encoder.pkl")   # Encoder untuk kategori menu
le_cat = joblib.load("profitability_encoder.pkl")    # Encoder target

# =======================
# App Configuration
# =======================
st.set_page_config(page_title="Menu Profitability Predictor", page_icon="🍽", layout="centered")

st.title("🍽 Restaurant Menu Profitability Predictor")
st.write("Masukkan kategori menu untuk memprediksi tingkat profitabilitasnya.")

# =======================
# Input
# =======================
menu_categories = list(le_menu.classes_)  # Ambil daftar kategori asli dari encoder
menu_category = st.selectbox("Pilih Kategori Menu:", menu_categories)

# Tombol prediksi
if st.button("Prediksi"):
    try:
        # Encode kategori menu
        menu_encoded = le_menu.transform([menu_category])[0]

        # Prediksi
        prediction_num = model.predict([[menu_encoded]])[0]
        prediction_label = le_cat.inverse_transform([prediction_num])[0]

        # Output hasil prediksi
        st.subheader("Hasil Prediksi:")
        if prediction_label == "High":
            st.success(f"✅ Profitabilitas: **{prediction_label}**")
        elif prediction_label == "Medium":
            st.info(f"ℹ Profitabilitas: **{prediction_label}**")
        else:
            st.error(f"⚠ Profitabilitas: **{prediction_label}**")

        st.write(f"*Kategori Menu:* {menu_category}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
