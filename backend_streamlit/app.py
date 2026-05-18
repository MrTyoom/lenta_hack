# app.py
import os
import json
import tempfile
import pandas as pd
import streamlit as st
from pathlib import Path
import base64
from io import BytesIO
import cv2

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

# Загрузка классов отделов
CLASS_NAMES_PATH = Path(__file__).parent.parent / "DEPARTMENT_CLASSIFICATION" / "train_model" / "models" / "class_names.json"
with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    ALL_DEPARTMENTS = json.load(f)
ALL_DEPARTMENTS = sorted(ALL_DEPARTMENTS) + ["Не определён", "Свой вариант"]

# Инициализация session_state
if 'report_df' not in st.session_state:
    st.session_state.report_df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "upload"
if 'custom_dept' not in st.session_state:
    st.session_state.custom_dept = ""
if 'bad_crops_data' not in st.session_state:
    st.session_state.bad_crops_data = None

# Авто-переключение на аналитику после обработки
if st.session_state.processing_done and st.session_state.report_df is not None:
    st.session_state.current_page = "analytics"

# Вкладки
tabs = st.tabs(["📹 Загрузка видео", "📊 Аналитика"])

def encode_crop_to_base64(crop):
    """Конвертировать numpy array crop в base64 для отображения"""
    if crop is None:
        return None
    _, buffer = cv2.imencode('.png', crop)
    return base64.b64encode(buffer).decode('utf-8')

# Вкладка 1: Загрузка видео
with tabs[0]:
    st.title("🤖 Система инспекции ценников")
    
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
            st.session_state.bad_crops_data = None
        
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
                
                # Сохранить данные для аналитики
                df = pd.DataFrame(domain_tags)
                
                # Извлечь crop изображения для плохих кропов
                bad_crops = df[df['SYS_trash'] == True].copy()
                bad_crops_data = []
                for _, row in bad_crops.iterrows():
                    if row.get('crop_image') is not None:
                        bad_crops_data.append({
                            'track_id': row['SYS_track_id'],
                            'confidence': row['SYS_quality_confidence'],
                            'image_base64': encode_crop_to_base64(row['crop_image']),
                            'color': row['color'],
                        })
                
                st.session_state.bad_crops_data = bad_crops_data
                
                # Удалить crop_image из DataFrame перед сохранением (не сериализуется)
                if 'crop_image' in df.columns:
                    df = df.drop(columns=['crop_image'])
                
                st.session_state.report_df = df
                st.session_state.metrics = domain_stats
                st.session_state.processing_done = True
                st.session_state.current_page = "analytics"
                st.rerun()
    
    # Подсказка
    st.info("💡 После завершения обработки вы будете автоматически перенаправлены на вкладку 'Аналитика'")
    
    # Кнопка быстрого перехода после обработки
    if st.session_state.processing_done and st.session_state.report_df is not None:
        st.success("✅ Обработка завершена!")
        if st.button("📊 Перейти к аналитике", use_container_width=True, key="go_to_analytics_btn"):
            st.session_state.current_page = "analytics"
            st.rerun()

# Вкладка 2: Аналитика
with tabs[1]:
    st.title("📊 Аналитика и результаты")
    
    # Проверка: есть ли данные для анализа
    if st.session_state.report_df is None or st.session_state.report_df.empty:
        st.warning("⚠️ Нет данных для отображения. Загрузите видео на вкладке 'Загрузка видео'.")
        
        # Показать исторические данные если есть
        all_sessions = analytics_use_case.repository.get_all_sessions()
        if all_sessions:
            st.markdown("---")
            st.subheader("📈 Историческая статистика")
            sessions_df = pd.DataFrame(all_sessions)
            st.dataframe(sessions_df[['video_filename', 'tag', 'mode_department', 'total_detections', 'created_at']])
            
            dept_dist = analytics_use_case.repository.get_department_distribution()
            if dept_dist:
                st.subheader("Распределение по отделам")
                dept_dist_df = pd.DataFrame(dept_dist)
                st.bar_chart(dept_dist_df.set_index('department')['session_count'])
        st.stop()
    
    df = st.session_state.report_df
    
    # Метрики
    st.subheader("📊 Метрики обработки")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Кадров", st.session_state.metrics.total_frames)
    m2.metric("Кропов (top-1)", st.session_state.metrics.total_detections)
    m3.metric("Отдел", f"{st.session_state.metrics.mode_department[:15]}... ({st.session_state.metrics.mode_department_count})")
    m4.metric("Время (сек)", st.session_state.metrics.elapsed_time)
    
    # Статистика качества
    total_crops = len(df)
    bad_crops_count = len(df[df['SYS_trash'] == True]) if 'SYS_trash' in df.columns else 0
    good_crops_count = total_crops - bad_crops_count
    bad_percentage = (bad_crops_count / total_crops * 100) if total_crops > 0 else 0
    
    m5.metric("🗑️ Мусор", f"{bad_crops_count} ({bad_percentage:.1f}%)", delta=f"-{good_crops_count} хороших" if bad_crops_count > 0 else "✓")
    
    # Статистика качества - подробная
    st.markdown("---")
    st.subheader("🏆 Качество кропов")
    
    q1, q2, q3 = st.columns(3)
    with q1:
        st.metric("Хорошие кропы", good_crops_count, delta=f"{100-bad_percentage:.1f}%")
    with q2:
        st.metric("Плохие кропы (мусор)", bad_crops_count, delta=f"{bad_percentage:.1f}%", delta_color="inverse")
    with q3:
        avg_quality_conf = df['SYS_quality_confidence'].mean() if 'SYS_quality_confidence' in df.columns else 0
        st.metric("Средняя уверенность качества", f"{avg_quality_conf:.1f}%")
    
    # График качества
    quality_chart_df = pd.DataFrame({
        'Статус': ['Хорошие', 'Мусор'],
        'Количество': [good_crops_count, bad_crops_count]
    })
    st.bar_chart(quality_chart_df.set_index('Статус'))
    
    # Статистика по цветам
    st.markdown("---")
    st.subheader("🎨 Распределение по цветам")
    
    if 'color' in df.columns:
        color_counts = df['color'].value_counts()
        color_df = color_counts.reset_index()
        color_df.columns = ['color', 'count']
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.bar_chart(color_df.set_index('color'))
        with c2:
            st.dataframe(color_df, hide_index=True)
    
    # Переопределение отдела
    st.markdown("---")
    st.subheader("🔧 Переопределить отдел")
    
    most_common_dept = st.session_state.metrics.mode_department
    
    try:
        default_index = ALL_DEPARTMENTS.index(most_common_dept) if most_common_dept in ALL_DEPARTMENTS else 0
    except:
        default_index = 0
    
    selected_dept = st.selectbox(
        "Выберите отдел из списка",
        options=ALL_DEPARTMENTS,
        index=default_index
    )
    
    if selected_dept == "Свой вариант":
        custom_dept = st.text_input("Введите название отдела", value=st.session_state.custom_dept)
        if custom_dept:
            st.session_state.custom_dept = custom_dept
            final_dept = custom_dept
        else:
            final_dept = most_common_dept
    else:
        final_dept = selected_dept
    
    if final_dept != most_common_dept:
        st.session_state.report_df['department'] = final_dept
        st.success(f"✅ Отдел изменён на **{final_dept}**")
    
    # Плохие кропы - визуализация
    if st.session_state.bad_crops_data and len(st.session_state.bad_crops_data) > 0:
        st.markdown("---")
        st.subheader(f"🗑️ Плохие кропы ({len(st.session_state.bad_crops_data)} шт.)")
        
        n_cols = 4
        n_rows = (len(st.session_state.bad_crops_data) + n_cols - 1) // n_cols
        
        for i, crop_data in enumerate(st.session_state.bad_crops_data):
            col_idx = i % n_cols
            if col_idx == 0:
                cols = st.columns(n_cols)
            
            with cols[col_idx]:
                st.image(
                    f"data:image/png;base64,{crop_data['image_base64']}",
                    caption=f"ID: {crop_data['track_id']}\nМусор: {crop_data['confidence']:.1f}%\nЦвет: {crop_data['color']}",
                    use_container_width=True
                )
    
    # Скачать CSV
    st.markdown("---")
    st.subheader("📥 Скачать результат")
    
    # Удалить технические колонки перед скачиванием
    cols_to_drop = [c for c in ['SYS_trash', 'SYS_quality_confidence', 'crop_image'] if c in df.columns]
    download_df = df.drop(columns=cols_to_drop) if cols_to_drop else df
    
    csv_bytes = download_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 Скачать CSV файл",
        data=csv_bytes,
        file_name=f"result_{st.session_state.last_uploaded_file.replace('.mp4', '.csv') if st.session_state.last_uploaded_file else 'result.csv'}",
        mime="text/csv",
        use_container_width=True
    )
    
    # Графики
    st.markdown("---")
    st.subheader("📈 Детекции по кадрам")
    detections_per_frame = df.groupby('frame_timestamp').size().reset_index(name='count')
    st.line_chart(detections_per_frame, x='frame_timestamp', y='count')
    
    st.subheader("🏷️ Распределение по отделам")
    dept_counts = df['department'].value_counts()
    dept_chart_df = dept_counts.reset_index()
    dept_chart_df.columns = ['department', 'count']
    st.bar_chart(dept_chart_df, x='department', y='count')
    
    # Историческая статистика
    st.markdown("---")
    st.subheader("📈 Историческая статистика")
    
    all_sessions = analytics_use_case.repository.get_all_sessions()
    if all_sessions:
        sessions_df = pd.DataFrame(all_sessions)
        st.dataframe(sessions_df[['video_filename', 'tag', 'mode_department', 'total_detections', 'created_at']])
        
        dept_dist = analytics_use_case.repository.get_department_distribution()
        if dept_dist:
            st.subheader("Распределение по отделам (все сессии)")
            dept_dist_df = pd.DataFrame(dept_dist)
            st.bar_chart(dept_dist_df.set_index('department')['session_count'])
    
    norm_count, prob_count = analytics_use_case.repository.get_problematic_stats()
    if norm_count > 0 or prob_count > 0:
        st.subheader("⚖️ Качество ценников")
        chart_df_1 = pd.DataFrame({'Статус': ['Корректные', 'Проблемные'], 'Количество': [norm_count, prob_count]})
        st.bar_chart(data=chart_df_1, x='Статус', y='Количество')
