import streamlit as st
import pandas as pd
import joblib

# Memuat model pipeline yang sudah disimpan
@st.cache_resource
def load_model():
    return joblib.load('model_knn_priority.pkl')

model = load_model()

st.set_page_config(page_title="Prediksi Order Priority", page_icon="📦", layout="centered")
st.title("🎯 Kalkulator Estimasi Order Priority")
st.markdown("Masukkan data operasional logistik di bawah ini untuk memprediksi tingkat urgensi pesanan.")
st.write("---")

ship_mode = st.selectbox("Mode Pengiriman (Ship Mode)", ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
segment = st.selectbox("Segmen Pelanggan (Segment)", ['Consumer', 'Corporate', 'Home Office'])
quantity = st.number_input("Jumlah Barang (Quantity)", min_value=1, max_value=100, value=3)
sales = st.number_input("Nilai Penjualan (Sales) dalam $", min_value=0.0, value=150.0)
profit = st.number_input("Keuntungan (Profit) dalam $", value=30.0)
shipping_cost = st.number_input("Biaya Ongkir (Shipping Cost) dalam $", min_value=0.0, value=15.0)

st.write("---")

if st.button("🔮 Hitung Prediksi Model KNN", type="primary"):
    input_data = pd.DataFrame([{
        'Ship Mode': ship_mode,
        'Shipping Cost': shipping_cost,
        'Quantity': quantity,
        'Sales': sales,
        'Profit': profit,
        'Segment': segment
    }])

    hasil_prediksi = model.predict(input_data)[0]

    if hasil_prediksi == 'Critical':
        st.error(f"Hasil Estimasi: **{hasil_prediksi.upper()}** (Segera Proses!)")
    elif hasil_prediksi == 'High':
        st.warning(f"Hasil Estimasi: **{hasil_prediksi.upper()}** (Prioritas Tinggi)")
    elif hasil_prediksi == 'Medium':
        st.info(f"Hasil Estimasi: **{hasil_prediksi.upper()}** (Prioritas Normal)")
    else:
        st.success(f"Hasil Estimasi: **{hasil_prediksi.upper()}** (Prioritas Rendah)")
