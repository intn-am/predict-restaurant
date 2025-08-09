import streamlit as st
import joblib

# =======================
# Load Model & Encoders
# =======================
@st.cache_resource
def load_all():
    model = joblib.load("best_random_forest_model.pkl")
    scaler = joblib.load("price_scaler.pkl")
    le_menu = joblib.load("menu_category_encoder.pkl")
    le_cat = joblib.load("profitability_encoder.pkl")
    return model, scaler, le_menu, le_cat

model, price_scaler, le_menu, le_cat = load_all()

# =======================
# App Configuration
# =======================
st.set_page_config(
    page_title="Menu Profitability Predictor",
    page_icon="🍽",
    layout="wide"
)

# =======================
# Custom CSS
# =======================
st.markdown("""
    <style>
    /* Background Image */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1541544741938-0af808871cc0');
        background-size: cover;
        background-attachment: fixed;
    }

    /* Overlay */
    .main-overlay {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 30px;
        border-radius: 15px;
    }

    .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #FF4B4B;
    }

    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #333;
    }

    .prediction-box {
        border-radius: 15px;
        padding: 25px;
        margin-top: 25px;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# =======================
# Overlay Container
# =======================
with st.container():
    st.markdown('<div class="main-overlay">', unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-title">🍽 Menu Profitability Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Prediksi profitabilitas menu restoran Anda menggunakan <b>Tuned Random Forest</b></p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("ℹ Tentang Aplikasi")
        st.write("""
            Aplikasi ini memprediksi **profitabilitas menu** berdasarkan:
            - **Kategori Menu** (contoh: Main Course, Beverages, dll)
            - **Harga Menu** dalam USD

            Manfaatkan untuk pengambilan keputusan harga dan kategori menu Anda!
        """)
        st.success("💡 Tip: Gunakan harga aktual agar hasil lebih akurat.")
        st.markdown("---")
        st.write("© 2025 | Dibuat dengan menggunakan **Streamlit** & **Random Forest** oleh Intan Aulia M ❤️")

    # Input Form
    st.subheader("📥 Masukkan Data Menu")
    col1, col2 = st.columns(2)

    with col1:
        menu_categories = list(le_menu.classes_)
        menu_category = st.selectbox("📋 Pilih Kategori Menu:", menu_categories, help="Kategori umum menu seperti Beverages, Main Course, dll")

    with col2:
        price = st.number_input("💲 Harga Menu (USD):", min_value=0.0, step=0.01, help="Masukkan harga menu dalam USD")

    # Prediction Button
    if st.button("🚀 Prediksi Profitabilitas", use_container_width=True):
        with st.spinner("⏳ Sedang memproses prediksi..."):
            try:
                # Preprocessing
                menu_encoded = le_menu.transform([menu_category])[0]
                price_scaled = price_scaler.transform([[price]])[0][0]

                # Prediction
                prediction_num = model.predict([[menu_encoded, price_scaled]])[0]
                prediction_label = le_cat.inverse_transform([prediction_num])[0]

                # Output
                st.markdown("### 📊 Hasil Prediksi:")

                if prediction_label == "High":
                    st.markdown(f"""
                        <div class="prediction-box" style="background-color:#D4EDDA; color:#155724;">
                        ✅ <b>Profitabilitas Tinggi</b><br>
                        Menu ini berpotensi memberikan <b>keuntungan besar</b> bagi restoran 🍾
                        </div>
                    """, unsafe_allow_html=True)

                elif prediction_label == "Medium":
                    st.markdown(f"""
                        <div class="prediction-box" style="background-color:#FFF3CD; color:#856404;">
                        ℹ <b>Profitabilitas Sedang</b><br>
                        Menu ini memberikan keuntungan pada tingkat <b>menengah</b>. Mungkin bisa dipertimbangkan promosi tambahan 🎯
                        </div>
                    """, unsafe_allow_html=True)

                else:
                    st.markdown(f"""
                        <div class="prediction-box" style="background-color:#F8D7DA; color:#721C24;">
                        ⚠ <b>Profitabilitas Rendah</b><br>
                        Menu ini cenderung memiliki <b>keuntungan rendah</b>. Coba evaluasi harga atau bahan baku 🧾
                        </div>
                    """, unsafe_allow_html=True)

                # Detail input
                st.markdown("#### 📌 Detail Input")
                st.write(f"- **Kategori Menu:** {menu_category}")
                st.write(f"- **Harga (USD):** ${price:.2f}")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {e}")

    st.markdown("</div>", unsafe_allow_html=True)
