import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="PersonaFinder", page_icon="🎓", layout="wide")
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1544604922-9543e46c53e3?q=80&w=1974&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    [data-testid="stVerticalBlock"] .st-emotion-cache-1r4qj8v {
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 2rem;
    }
    
    [data-testid="stSidebar"] {
        background: none;
    }

    .main {
        font-family: 'Poppins', sans-serif; 
    }

    .stButton>button {
        background-color: #4B0082;
        color: white;
        border-radius: 20px;
        border: 1px solid #4B0082;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        transition: all 0.3s ease-in-out;
        display: block;
        margin: 0 auto;
    }
    .stButton>button:hover {
        background-color: white;
        color: #4B0082;
        border: 1px solid #4B0082;
    }

    .result-box {
        border: 2px solid #4B0082;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-top: 20px;
        background-color: #F0F2F6;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        kmeans_model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return kmeans_model, scaler
    except FileNotFoundError:
        st.error("File model ('kmeans_model.pkl' atau 'scaler.pkl') tidak ditemukan. Pastikan file berada di folder yang sama.")
        return None, None

kmeans, scaler = load_model()

persona_map = {
    0: 'Mahasiswa Unggulan',
    1: 'Mahasiswa Penyimak',
    2: 'Mahasiswa Vokal',
    3: 'Mahasiswa Digital'
}

persona_icons = {
    'Mahasiswa Unggulan': '🏆',
    'Mahasiswa Penyimak': '🎧',
    'Mahasiswa Vokal': '🗣️',
    'Mahasiswa Digital': '💻'
}

persona_desc = {
    'Mahasiswa Unggulan': 'Selamat, Anda adalah **Mahasiswa Unggulan**! Anda menaklukkan dunia akademik dari segala sisi—tugas yang brilian, diskusi yang berbobot, dan penguasaan platform digital. Anda adalah inspirasi. Terus pertahankan momentum luar biasa ini!',
    'Mahasiswa Penyimak': 'Anda adalah **Mahasiswa Penyimak**, pilar kelas yang konsisten dan dapat diandalkan. Kehadiran dan ketekunan Anda adalah fondasi yang kuat. Kini, lepaskan suara Anda dalam diskusi—satu langkah kecil yang akan membuka potensi terbesar Anda!',
    'Mahasiswa Vokal': 'Dengan kepercayaan diri seorang **Mahasiswa Vokal**, ide-ide Anda selalu menyemarakkan diskusi. Keberanian Anda adalah aset yang langka! Untuk melengkapi bakat ini, alokasikan fokus ekstra pada tugas dan kehadiran, maka Anda tak akan terhentikan.',
    'Mahasiswa Digital': 'Anda adalah **Mahasiswa Digital**, seorang navigator ulung di lautan informasi online. Kemampuan Anda untuk belajar mandiri melalui teknologi adalah tiket emas untuk masa depan. Keterampilan ini sangat dicari di dunia modern.'
}

with st.sidebar:
    st.title("📖 Tentang Persona Belajar")
    st.info(
        """
        Setiap mahasiswa memiliki cara belajar yang unik. Model ini membantu Anda memahami pola perilaku Anda dengan mengelompokkannya ke dalam salah satu dari empat persona utama:
        - **🏆 Mahasiswa Unggulan:** Seimbang dan berprestasi di semua bidang.
        - **🎧 Mahasiswa Penyimak:** Rajin dan patuh, namun cenderung pasif.
        - **🗣️ Mahasiswa Vokal:** Percaya diri dalam diskusi, namun perlu fokus pada tugas.
        - **💻 Mahasiswa Digital:** Ahli belajar mandiri menggunakan teknologi.
        
        Kenali diri Anda dan temukan cara terbaik untuk berkembang!
        """
    )


# --- Tampilan Web  ---
if kmeans and scaler:
    st.title("🎓 PersonaFinder")
    st.markdown("### Kenali Gaya Belajarmu, Maksimalkan Potensimu!")
    st.write("---")

    with st.container():
        st.subheader("Masukkan Aktivitas Akademik Anda")
        
        kehadiran = st.slider('Kehadiran di Kelas (%)', 0, 100, 75)
        diskusi = st.slider('Partisipasi Diskusi (skor)', 0, 100, 70)
        tugas = st.slider('Nilai Rata-rata Tugas', 0, 100, 80)
        elearning = st.slider('Aktivitas di E-Learning (skor)', 0, 100, 75)
        
        st.write("")
        
        if st.button('✨ Prediksi Persona Saya'):
            input_data = np.array([[kehadiran, diskusi, tugas, elearning]])
            input_scaled = scaler.transform(input_data)
            
            pred_cluster = kmeans.predict(input_scaled)[0]
            pred_persona_name = persona_map[pred_cluster]
            pred_persona_desc = persona_desc[pred_persona_name]
            pred_persona_icon = persona_icons[pred_persona_name]
            
            st.markdown(f"""
            <div class="result-box">
                <p style="font-size: 24px;">Persona Belajar Anda adalah:</p>
                <h2 style="color: #4B0082;">{pred_persona_icon} {pred_persona_name}</h2>
                <p style="font-size: 18px; text-align: justify;">{pred_persona_desc}</p>
            </div>
            """, unsafe_allow_html=True)
