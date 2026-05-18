# app.py
import os
import tempfile
import pandas as pd
import streamlit as st
from pathlib import Path

from application import ProcessVideoUseCase, GetAnalyticsUseCase, ProgressListener
from infrastructure import YoloObjectDetector, OpenCvVideoReader, SqliteSessionRepository

class StreamlitProgressAdapter(ProgressListener):
    def __init__(self, bar_element, text_element):
        self.bar = bar_element
        self.text = text_element

    def on_progress(self, processed: int, total: int, detections: int):
        if total > 0:
            percent = min(processed / total, 1.0)
            self.bar.progress(percent)
            self.text.text(f"Кадры: {processed} / {total} ({int(percent * 100)}%) | Треков: {detections}")

st.set_page_config(page_title="Lenta Production AI", layout="wide", page_icon="🤖")
st.title("🤖 Система инспекции ценников")

MODEL_PATH = str(Path(__file__).parent.parent / "MAIN_MODULE" / "models" / "best.pt")

@st.cache_resource
def init_layers():
    repo = SqliteSessionRepository()
    detector = YoloObjectDetector(model_path=MODEL_PATH)
    reader = OpenCvVideoReader()
    return detector, reader, repo

st.sidebar.header("🖥️ Мониторинг системы")
try:
    detector_infra, video_infra, repo_infra = init_layers()
    st.sidebar.success("🟢 GPU/CPU и YOLO подключены.")
except Exception as e:
    st.sidebar.error("🔴 Ошибка оборудования")
    st.error(f"Ошибка инициализации: {e}")
    st.stop()

video_use_case = ProcessVideoUseCase(detector_infra, video_infra, repo_infra)
analytics_use_case = GetAnalyticsUseCase(repo_infra)

# Инициализация session_state
if 'report_df' not in st.session_state:
    st.session_state.report_df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

uploaded_file = st.file_uploader("Шаг 1: Загрузите видео", type=["mp4", "avi", "mov"])

rotation_options = ["90° против часовой", "90° по часовой", "180°", "0° (без поворота)"]
selected_rotation = st.selectbox("Шаг 2: Ориентация видео", options=rotation_options, index=0)

if uploaded_file is not None:
    # Проверка на новый файл
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.processing_done = False
        st.session_state.report_df = None
        st.session_state.metrics = None
    
    # Авто-запуск обработки
    if not st.session_state.processing_done:
        with st.spinner("Обработка видео..."):
            ui_progress_bar = st.progress(0)
            ui_status_text = st.empty()
            
            progress_adapter = StreamlitProgressAdapter(ui_progress_bar, ui_status_text)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_video_path = temp_file.name
            
            domain_tags, domain_stats = video_use_case.execute(
                video_path=temp_video_path,
                rotation=selected_rotation,
                session_tag=uploaded_file.name,
                progress_listener=progress_adapter
            )
            
            os.unlink(temp_video_path)
            
            st.session_state.report_df = pd.DataFrame(domain_tags)
            st.session_state.metrics = domain_stats
            st.session_state.processing_done = True
            st.rerun()

# Показ результатов
if st.session_state.report_df is not None and not st.session_state.report_df.empty:
    st.markdown("---")
    st.header("📊 Результаты")
    
    # Метрики
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Кадров", st.session_state.metrics.total_frames)
    m2.metric("Кропов (top-1)", st.session_state.metrics.total_detections)
    m3.metric("Отдел", f"{st.session_state.metrics.mode_department[:20]}... ({st.session_state.metrics.mode_department_count})")
    m4.metric("Время (сек)", st.session_state.metrics.elapsed_time)
    
    # Теги отделов (цветные)
    st.subheader("🏷️ Отделы в видео")
    dept_counts = st.session_state.report_df['department'].value_counts()
    dept_colors = {
        "Алкоголь": "#FF6B6B",
        "Молоко": "#4ECDC4",
        "Хлеб": "#FFE66D",
        "Кондитерское": "#95E1D3",
        "Мясо": "#F38181",
        "Рыба": "#AA96DA",
        "Овощи/Фрукты": "#FCBAD3",
        "Бакалея": "#FFD93D",
        "Другое": "#6BCB77",
        "Не определён": "#808080"
    }
    
    num_tags = len(dept_counts)
    if num_tags > 0:
        tag_cols = st.columns(min(num_tags, 4))
        for i, (dept, count) in enumerate(dept_counts.items()):
            color = dept_colors.get(dept, "#808080")
            with tag_cols[i % len(tag_cols)]:
                st.markdown(
                    f"""<div style="background-color: {color}; color: white; padding: 10px 20px; border-radius: 20px; text-align: center; font-weight: bold; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                    {dept}<br/><small style="font-size: 12px">{count} кропов</small></div>""",
                    unsafe_allow_html=True
                )
    
    # Переопределение отдела
    st.markdown("**🔧 Переопределить отдел:**")
    most_common_dept = st.session_state.metrics.mode_department
    selected_dept = st.selectbox(
        "Выберите отдел",
        options=dept_counts.index.tolist(),
        index=0,
        format_func=lambda x: f"{x} ({dept_counts[x]} кропов)"
    )
    
    if selected_dept != most_common_dept:
        st.session_state.report_df['department'] = selected_dept
        st.success(f"✅ Отдел изменён на **{selected_dept}**")
    
    # Скачать CSV
    st.subheader("📥 Скачать результат")
    csv_bytes = st.session_state.report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Скачать файл формата CSV",
        data=csv_bytes,
        file_name=f"result_{st.session_state.last_uploaded_file}",
        mime="text/csv",
        use_container_width=True
    )
    
    # Графики
    st.markdown("---")
    st.subheader("📈 Детекции по кадрам")
    detections_per_frame = st.session_state.report_df.groupby('frame_timestamp').size().reset_index(name='count')
    st.line_chart(detections_per_frame, x='frame_timestamp', y='count')
    
    st.subheader("🏷️ Распределение по отделам")
    dept_chart_df = dept_counts.reset_index()
    dept_chart_df.columns = ['department', 'count']
    st.bar_chart(dept_chart_df, x='department', y='count')

# Исторический дашборд
st.markdown("---")
st.header("📈 Историческая статистика")

all_sessions = analytics_use_case.repository.get_all_sessions()
if all_sessions:
    sessions_df = pd.DataFrame(all_sessions)
    st.subheader("Все сессии")
    st.dataframe(sessions_df[['video_filename', 'tag', 'mode_department', 'total_detections', 'created_at']])
    
    dept_dist = analytics_use_case.repository.get_department_distribution()
    if dept_dist:
        st.subheader("Распределение по отделам")
        dept_dist_df = pd.DataFrame(dept_dist)
        st.bar_chart(dept_dist_df.set_index('department')['session_count'])
else:
    st.info("База данных пока пуста. Загрузите видео для обработки.")

norm_count, prob_count = analytics_use_case.repository.get_problematic_stats()
if norm_count > 0 or prob_count > 0:
    st.subheader("⚖️ Качество ценников")
    chart_df_1 = pd.DataFrame({'Статус': ['Корректные', 'Проблемные'], 'Количество': [norm_count, prob_count]})
    st.bar_chart(data=chart_df_1, x='Статус', y='Количество', color="#ff4b4b")
