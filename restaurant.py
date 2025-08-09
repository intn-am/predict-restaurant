import streamlit as st
import joblib

# =======================
# Load Model & Encoders
# =======================
model = joblib.load("best_random_forest_model.pkl")
price_scaler = joblib.load("price_scaler.pkl")
le_menu = joblib.load("menu_category_encoder.pkl")
le_cat = joblib.load("profitability_encoder.pkl")

# =======================
# App Configuration
# =======================
st.set_page_config(page_title="Menu Profitability Predictor", page_icon="🍽", layout="centered")

st.title("🍽 Restaurant Menu Profitability Predictor")
st.write("Masukkan kategori menu dan harga untuk memprediksi tingkat profitabilitas menggunakan **Tuned Random Forest**.")

# =======================
# Input
# =======================
menu_categories = list(le_menu.classes_)
menu_category = st.selectbox("📋 Pilih Kategori Menu:", menu_categories)

price = st.number_input("💲 Harga Menu (USD):", min_value=0.0, step=0.01)

# Tombol prediksi
if st.button("🚀 Prediksi Profitabilitas"):
    try:
        # Encode kategori & scaling harga
        menu_encoded = le_menu.transform([menu_category])[0]
        price_scaled = price_scaler.transform([[price]])[0][0]

        # Prediksi
        prediction_num = model.predict([[menu_encoded, price_scaled]])[0]
        prediction_label = le_cat.inverse_transform([prediction_num])[0]

        # Output hasil prediksi
        st.subheader("📊 Hasil Prediksi:")
        if prediction_label == "High":
            st.success(f"✅ Profitabilitas: **{prediction_label}**")
        elif prediction_label == "Medium":
            st.info(f"ℹ Profitabilitas: **{prediction_label}**")
        else:
            st.error(f"⚠ Profitabilitas: **{prediction_label}**")

        st.write(f"*Kategori Menu:* {menu_category}")
        st.write(f"*Harga (USD):* ${price:.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
