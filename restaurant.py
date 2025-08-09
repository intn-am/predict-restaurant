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
# Custom CSS (Background Gambar + Header Box)
# =======================
st.markdown("""
    <style>
        .header-container {
            background-image: url('https://images.app.goo.gl/QkCxjro2JDYc6eaS6');
            background-size: cover;
            background-position: center;
            padding: 60px;
            border-radius: 15px;
            text-align: center;
            margin-bottom: 30px;
        }
        .header-title {
            background-color: rgba(255, 255, 255, 0.85);
            display: inline-block;
            padding: 20px 40px;
            border-radius: 12px;
        }
        .header-title h1 {
            font-size: 48px;
            color: #d62828;
            margin-bottom: 10px;
        }
        .header-title p {
            font-size: 20px;
            color: #333;
        }
    </style>

    <div class='header-container'>
        <div class='header-title'>
            <h1>🍽 Menu Profitability Predictor</h1>
            <p>Prediksi profitabilitas menu restoran menggunakan <b>Tuned Random Forest</b></p>
        </div>
    </div>
""", unsafe_allow_html=True)

# =======================
# Sidebar
# =======================
with st.sidebar:
    st.header("ℹ Tentang Aplikasi")
    st.write("""
        Aplikasi ini memprediksi **profitabilitas menu** berdasarkan:
        - **Kategori Menu** (Beverages, Main Course, Dessert, dll)
        - **Harga Menu** (dalam USD)

        Gunakan model ini untuk membantu strategi harga & kategori menu.
    """)
    st.markdown("---")
    st.write("© 2025 | Dibuat dengan ❤️ menggunakan **Streamlit** & **Random Forest** oleh Intan Aulia M")

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
                <div style="background-color:#D4EDDA; color:#155724; padding:20px; border-radius:10px;">
                ✅ <b>Profitabilitas Tinggi</b><br>
                Menu ini berpotensi memberikan <b>keuntungan besar</b> bagi restoran Anda.
                </div>
            """, unsafe_allow_html=True)

        elif prediction_label == "Medium":
            st.markdown(f"""
                <div style="background-color:#FFF3CD; color:#856404; padding:20px; border-radius:10px;">
                ℹ <b>Profitabilitas Sedang</b><br>
                Menu ini memberikan keuntungan pada tingkat <b>menengah</b>.
                </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown(f"""
                <div style="background-color:#F8D7DA; color:#721C24; padding:20px; border-radius:10px;">
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
