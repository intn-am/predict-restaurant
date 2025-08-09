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
st.set_page_config(page_title="Menu Profitability Predictor", page_icon="🍽", layout="wide")

# =======================
# Custom CSS
# =======================
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: #555;
    }
    .prediction-box {
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# =======================
# Header
# =======================
st.markdown('<p class="main-title">🍽 Restaurant Menu Profitability Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Prediksi profitabilitas menu restoran Anda dengan model **Tuned Random Forest**</p>', unsafe_allow_html=True)
st.markdown("---")

# =======================
# Sidebar
# =======================
with st.sidebar:
    st.header("ℹ Tentang Aplikasi")
    st.write("""
        Aplikasi ini memprediksi **profitabilitas menu** berdasarkan:
        - **Kategori Menu** (Beverages, Main Course, Dessert, dll)
        - **Harga Menu** (dalam USD)

        Gunakan model ini untuk membantu strategi harga & kategori menu Anda.
    """)
    st.markdown("---")
    st.write("© 2025 | Dibuat dengan ❤️ menggunakan **Streamlit** & **Random Forest**")

# =======================
# Input Form
# =======================
col1, col2 = st.columns(2)

with col1:
    menu_categories = list(le_menu.classes_)
    menu_category = st.selectbox("📋 Pilih Kategori Menu:", menu_categories)

with col2:
    price = st.number_input("💲 Harga Menu (USD):", min_value=0.0, step=0.01)

# =======================
# Prediction
# =======================
if st.button("🚀 Prediksi Profitabilitas", use_container_width=True):
    try:
        # Encode kategori & scaling harga
        menu_encoded = le_menu.transform([menu_category])[0]
        price_scaled = price_scaler.transform([[price]])[0][0]

        # Prediksi
        prediction_num = model.predict([[menu_encoded, price_scaled]])[0]
        prediction_label = le_cat.inverse_transform([prediction_num])[0]

        # Output
        st.markdown("### 📊 Hasil Prediksi:")

        if prediction_label == "High":
            st.markdown(f"""
                <div class="prediction-box" style="background-color:#D4EDDA; color:#155724;">
                ✅ <b>Profitabilitas Tinggi</b><br>
                Menu ini berpotensi memberikan <b>keuntungan besar</b> bagi restoran Anda.
                </div>
            """, unsafe_allow_html=True)

        elif prediction_label == "Medium":
            st.markdown(f"""
                <div class="prediction-box" style="background-color:#FFF3CD; color:#856404;">
                ℹ <b>Profitabilitas Sedang</b><br>
                Menu ini memberikan keuntungan pada tingkat <b>menengah</b>.
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
                <div class="prediction-box" style="background-color:#F8D7DA; color:#721C24;">
                ⚠ <b>Profitabilitas Rendah</b><br>
                Menu ini cenderung memiliki <b>keuntungan rendah</b>.
                </div>
            """, unsafe_allow_html=True)

        # Detail input
        st.markdown("#### 📌 Detail Input")
        st.write(f"- **Kategori Menu:** {menu_category}")
        st.write(f"- **Harga (USD):** ${price:.2f}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
