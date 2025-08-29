import streamlit as st
import requests
import io
import tempfile
import time

st.set_page_config(page_title="AI Depression", layout="wide")

# ================= 세션 상태 =================
if "uploaded_video_data" not in st.session_state:
    st.session_state.uploaded_video_data = None
if "video_uploaded_time" not in st.session_state:
    st.session_state.video_uploaded_time = None
if "video_analysis_done" not in st.session_state:
    st.session_state.video_analysis_done = False
if "realtime_analysis_done" not in st.session_state:
    st.session_state.realtime_analysis_done = False

# ================= CSS =================
st.markdown("""
<style>
.stApp { font-family: "Neue Helvetica", Helvetica, Arial, sans-serif; }
[data-testid="stSidebar"] { background-color: white; color: black; font-family: "Neue Helvetica", Helvetica, Arial, sans-serif; }
[data-testid="stSidebar"] h1 { text-align: center; }
[data-testid="stSidebar"] img { max-width: 100%; display: block; margin: 0 auto; }
.stButton > button { color: white !important; background-color: #4A90E2; height: 60px; margin-top: 12px; font-family: "Neue Helvetica", Helvetica, Arial, sans-serif; font-size: 16px; border-radius: 8px; }
iframe[title="Streamlit Video"] { height: 625px !important; }
.result-card { background-color: #f9f9f9; border: 2px solid #ddd; border-radius: 15px; padding: 20px; margin-top: 20px; font-family: "Neue Helvetica", Helvetica, Arial, sans-serif; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
.result-title { font-size: 20px; font-weight: bold; color: #333; margin-bottom: 10px; }
.result-item { font-size: 16px; color: #555; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ================= 사이드바 =================
with st.sidebar:
    st.title("AI Depression")
    st.image("C:\\Users\\user\\Downloads\\my_png.jpg", use_container_width=True)

# ================= 탭 생성 =================
tab1, tab2 = st.tabs(["📂 멀티미디어 업로드 & AI 분석", "🎤 실시간 분석"])

# ================= TAB 1 =================
with tab1:
    st.header("멀티미디어 업로드 & AI 분석")
    col1, col2 = st.columns([3, 2])

    # 왼쪽: 동영상 플레이어
    with col1:
        if st.session_state.uploaded_video_data is not None:
            st.video(st.session_state.uploaded_video_data, format='video/mp4', start_time=0)
        else:
            st.video(io.BytesIO(), format='video/mp4')

    # 오른쪽: 업로드 + 자동 분석 상태
    with col2:
        new_uploaded_video = st.file_uploader(
            "동영상 업로드", type=["mp4", "mov", "avi"], key="video_uploader_tab1"
        )

        if new_uploaded_video is not None:
            if st.session_state.uploaded_video_data != new_uploaded_video:
                st.session_state.uploaded_video_data = new_uploaded_video
                st.session_state.video_uploaded_time = time.time()
                st.session_state.video_analysis_done = False

        response_placeholder = st.empty()

        # 자동 분석 40초 카운트다운
        if st.session_state.uploaded_video_data and not st.session_state.video_analysis_done:
            elapsed = time.time() - st.session_state.video_uploaded_time
            if elapsed < 20:
                remaining = int(20 - elapsed)
                response_placeholder.info("자동으로 AI 분석을 시작합니다...")
            else:
                response_placeholder.info("📡 AI 분석 서버로 데이터 전송 중...")
                image_path = "./data/sample.png"
                audio_path = "./data/sample.wav"
                files = {}
                try:
                    files["image"] = ("sample.png", open(image_path, "rb"), "image/png")
                    files["audio"] = ("sample.wav", open(audio_path, "rb"), "audio/wav")
                except FileNotFoundError:
                    response_placeholder.error("❌ 지정된 경로의 이미지 또는 오디오 파일을 찾을 수 없습니다.")
                    files = None

                if files:
                    try:
                        FASTAPI_URL = "http://localhost:8999/predict"
                        response = requests.post(FASTAPI_URL, files=files)
                        if response.status_code == 200:
                            result_json = response.json()
                            # 결과 카드 출력
                            items_html = f"""
                            <div class="result-item"><b>최종 진단:</b> {result_json['final_prediction']}</div>
                            <div class="result-item"><b>우울증 가능성:</b> {result_json['depression_percentage']}%</div>
                            <div class="result-item"><b>위험도:</b> {result_json['risk_level']}</div>
                            <div class="result-item"><b>개별 결과:</b></div>
                            <ul>
                                <li>이미지: {result_json['individual_results']['image']['percentage']}%</li>
                                <li>음성: {result_json['individual_results']['sound']['percentage']}%</li>
                                <li>텍스트: {result_json['individual_results']['text']['percentage']}%</li>
                            </ul>
                            """
                            st.markdown(f"<div class='result-card'>{items_html}</div>", unsafe_allow_html=True)
                            st.session_state.video_analysis_done = True
                        else:
                            response_placeholder.error(f"❌ 분석 실패: {response.status_code}")
                    except Exception as e:
                        response_placeholder.error(f"❌ 서버 연결 실패: {e}")

# ================= TAB 2 =================
with tab2:
    st.header("실시간 분석")
    st.write("📷 사진을 촬영하고 🎤 음성을 녹음한 뒤, 'AI 분석 시작' 버튼을 누르세요.")
    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    # 왼쪽: 카메라 + 음성 입력
    with col1:
        picture_widget = st.camera_input("카메라 촬영", label_visibility="visible")
        audio_widget = st.audio_input("음성 녹음", label_visibility="visible")

    # 오른쪽: 버튼 + 결과
    with col2:
        status_placeholder = st.empty()

        if st.button("🧠 AI 분석 시작", key="analyze_button"):
            if not picture_widget:
                st.warning("📷 사진을 먼저 촬영해주세요.")
            if not audio_widget:
                st.warning("🎤 음성을 먼저 녹음해주세요.")

            if picture_widget and audio_widget:
                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_img.write(picture_widget.getvalue())
                temp_img.flush()
                st.image(picture_widget, caption="촬영된 사진")

                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_audio.write(audio_widget.getvalue())
                temp_audio.flush()
                st.audio(audio_widget, format="audio/wav")

                status_placeholder.info("📡 AI 분석 서버로 데이터 전송 중...")

                try:
                    FASTAPI_URL = "http://localhost:8999/predict"
                    files_to_send = {
                        "image": ("realtime_image.png", open(temp_img.name, 'rb'), "image/png"),
                        "audio": ("realtime_audio.wav", open(temp_audio.name, 'rb'), "audio/wav")
                    }
                    response = requests.post(FASTAPI_URL, files=files_to_send)
                    if response.status_code == 200:
                        result_json = response.json()
                        items_html = f"""
                        <div class="result-item"><b>최종 진단:</b> {result_json['final_prediction']}</div>
                        <div class="result-item"><b>우울증 가능성:</b> {result_json['depression_percentage']}%</div>
                        <div class="result-item"><b>위험도:</b> {result_json['risk_level']}</div>
                        <div class="result-item"><b>개별 결과:</b></div>
                        <ul>
                            <li>이미지: {result_json['individual_results']['image']['percentage']}%</li>
                            <li>음성: {result_json['individual_results']['sound']['percentage']}%</li>
                            <li>텍스트: {result_json['individual_results']['text']['percentage']}%</li>
                        </ul>
                        """
                        st.markdown(f"<div class='result-card'>{items_html}</div>", unsafe_allow_html=True)
                        st.session_state.realtime_analysis_done = True
                        status_placeholder.success("🧠 AI 실시간 분석 완료")
                    else:
                        status_placeholder.error(f"❌ 분석 실패: {response.status_code} - {response.text}")
                except Exception as e:
                    status_placeholder.error(f"❌ 요청 중 오류 발생: {e}")
