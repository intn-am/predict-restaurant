import streamlit as st
import joblib

# =======================
# Load Model & Encoders
# =======================
model = joblib.load("best_random_forest_model.pkl")
price_scaler = joblib.load("price_scaler.pkl")
le_menu = joblib.load("menu_category_encoder.pkl")       # Encoder untuk MenuCategory
le_cat = joblib.load("profitability_encoder.pkl")        # Encoder untuk Profitability (target)

# =======================
# App Configuration
# =======================
st.set_page_config(
    page_title="Restaurant Profitability Predictor",
    page_icon="🍽",
    layout="wide"
)

# =======================
# Header
# =======================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
    }
    .sub-text {
        text-align: center;
        color: #666;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-title">🍽 Restaurant Menu Profitability Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Prediksi tingkat profitabilitas menu restoran berdasarkan kategori dan harga.</p>', unsafe_allow_html=True)
st.markdown("---")

# =======================
# Sidebar - Input Form
# =======================
with st.sidebar:
    st.header("🔧 Input Data Menu")

    # Menu Category Selection
    menu_categories = list(le_menu.classes_)
    menu_category = st.selectbox(
        "📋 Pilih Kategori Menu:",
        menu_categories
    )

    # Price Input
    price = st.number_input(
        "💲 Harga Menu (USD):",
        min_value=0.0,
        step=0.01,
        help="Masukkan harga menu dalam USD"
    )

    # Predict Button
    predict_button = st.button("🚀 Prediksi Profitabilitas", use_container_width=True)

# =======================
# Prediction
# =======================
if predict_button:
    try:
        # Encode kategori menu
        menu_encoded = le_menu.transform([menu_category])[0]

        # Scale harga
        price_scaled = price_scaler.transform([[price]])[0][0]

        # Prediksi
        prediction_num = model.predict([[menu_encoded, price_scaled]])[0]

        # Decode hasil prediksi menjadi label Profitability
        prediction_label = le_cat.inverse_transform([prediction_num])[0]

        # =======================
        # Display Result
        # =======================
        st.subheader("📊 Hasil Prediksi")

        if prediction_label == "High":
            st.success(f"✅ Predicted Profitability: *{prediction_label}*")
            st.markdown("💡 Menu ini berpotensi memberikan *keuntungan tinggi* bagi restoran. Pertahankan kualitas dan promosi!")
        elif prediction_label == "Medium":
            st.info(f"ℹ Predicted Profitability: *{prediction_label}*")
            st.markdown("📈 Menu ini memberikan *keuntungan sedang*. Pertimbangkan strategi harga atau paket promo untuk meningkatkan penjualan.")
        else:
            st.error(f"⚠ Predicted Profitability: *{prediction_label}*")
            st.markdown("🔍 Menu ini memiliki *keuntungan rendah*. Perlu evaluasi bahan baku, harga jual, atau strategi pemasaran.")

        st.markdown("---")
        st.markdown("### 📌 Detail Input")
        st.write(f"*Kategori Menu:* {menu_category}")
        st.write(f"*Harga (USD):* ${price:.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

# =======================
# Footer
# =======================
st.markdown("---")
st.caption("© 2025 Restaurant Profitability Predictor | Powered by Random Forest Classifier & Streamlit")
