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
        self.stage_names = [
            "1. Детекция ценников (YOLO)",
            "2. Оценка качества",
            "3. Определение цветов",
            "4. Классификация отделов",
            "5. VLM OCR распознавание",
            "6. Поиск товаров",
            "7. LLM подбор SKU"
        ]

    def on_progress(self, processed: int, total: int, detections: int, stage: int = 1, stage_name: str = "Детекция"):
        if total > 0:
            stage_percent = processed / total if total > 0 else 0
            overall_percent = ((stage - 1) * (1.0 / 7.0)) + (stage_percent * (1.0 / 7.0))
            self.bar.progress(min(overall_percent, 1.0))
            stage_status = "✓" if stage_percent >= 1.0 else "→"
            self.text.text(f"Этап {stage}/7: {stage_name} {stage_status} ({int(stage_percent * 100)}%)")


def encode_crop_to_base64(crop):
    if crop is None:
        return None
    _, buffer = cv2.imencode('.png', crop)
    return base64.b64encode(buffer).decode('utf-8')


st.set_page_config(page_title="Lenta Production AI", layout="wide", page_icon="🤖")

MODEL_PATH = str(Path(__file__).parent.parent / "MAIN_MODULE" / "models" / "best.pt")

@st.cache_resource
def init_layers():
    repo = SqliteSessionRepository()
    detector = YoloObjectDetector(model_path=MODEL_PATH)
    reader = OpenCvVideoReader()
    return detector, reader, repo

try:
    detector_infra, video_infra, repo_infra = init_layers()
except Exception as e:
    st.error(f"Ошибка инициализации: {e}")
    st.stop()

video_use_case = ProcessVideoUseCase(detector_infra, video_infra, repo_infra)
analytics_use_case = GetAnalyticsUseCase(repo_infra)

if 'report_df' not in st.session_state:
    st.session_state.report_df = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False
if 'bad_crops_data' not in st.session_state:
    st.session_state.bad_crops_data = None
if 'tag_value' not in st.session_state:
    st.session_state.tag_value = ""

if st.session_state.processing_done and st.session_state.report_df is not None:
    if not st.session_state.tag_value:
        st.session_state.tag_value = st.session_state.metrics.mode_department

st.title("🤖 Система инспекции ценников")

uploaded_file = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])

rotation_options = ["90° против часовой", "90° по часовой", "180°", "0° (без поворота)"]
selected_rotation = st.selectbox("Ориентация видео", options=rotation_options, index=0)
    
if uploaded_file is not None:
    if st.session_state.last_uploaded_file != uploaded_file.name:
        st.session_state.last_uploaded_file = uploaded_file.name
        st.session_state.processing_done = False
        st.session_state.report_df = None
        st.session_state.metrics = None
        st.session_state.bad_crops_data = None
        st.session_state.tag_value = ""
    
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
            
            df = domain_tags
            
            st.session_state.bad_crops_data = video_use_case.bad_crops_data
            
            st.session_state.report_df = df
            st.session_state.metrics = domain_stats
            st.session_state.tag_value = domain_stats.mode_department
            st.session_state.processing_done = True
            st.rerun()

st.info("💡 После завершения обработки ниже появится статистика")

if st.session_state.processing_done and st.session_state.report_df is not None:
    df = st.session_state.report_df
    
    st.markdown("---")
    st.subheader("📊 Статистика обработки")
    
    dept_input = st.text_input(
        "Отдел (определён автоматически)",
        value=st.session_state.tag_value,
        key="dept_tag_input"
    )
    if dept_input != st.session_state.tag_value:
        st.session_state.tag_value = dept_input
        st.session_state.report_df['department'] = dept_input
        st.success(f"✅ Отдел изменён на **{dept_input}**")
    
    total_crops = len(df)
    bad_crops_count = len(df[df['SYS_trash'] == True]) if 'SYS_trash' in df.columns else 0
    good_crops_count = total_crops - bad_crops_count
    
    matched_count = len(df[df['id_sku'].notna() & (df['id_sku'] != 'нет')]) if 'id_sku' in df.columns else 0
    ocr_count = len(df[df['product_name'].notna() & (df['product_name'] != '')]) if 'product_name' in df.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📦 Всего ценников", total_crops)
    with col2:
        st.metric("✅ Хороших", good_crops_count)
    with col3:
        st.metric("🗑️ Плохих", bad_crops_count)
    with col4:
        st.metric("🏷️ SKU найдено", matched_count)
    
    if 'product_name' in df.columns:
        good_df = df[df['SYS_trash'] == False] if 'SYS_trash' in df.columns else df
        if not good_df.empty:
            st.markdown("### 📋 Распознанные ценники")
            display_cols = ['product_name', 'price_card', 'barcode', 'id_sku', 'color', 'department']
            available_cols = [c for c in display_cols if c in good_df.columns]
            if available_cols:
                st.dataframe(good_df[available_cols].head(20), use_container_width=True)
    
    st.markdown("### 📥 Экспорт")
    csv_col, progress_col = st.columns([1, 2])
    
    with csv_col:
        cols_to_drop = [c for c in ['SYS_trash', 'SYS_quality_confidence'] if c in df.columns]
        download_df = df.drop(columns=cols_to_drop) if cols_to_drop else df
        csv_bytes = download_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Скачать CSV",
            data=csv_bytes,
            file_name=f"result_{st.session_state.last_uploaded_file.replace('.mp4', '.csv') if st.session_state.last_uploaded_file else 'result.csv'}",
            mime="text/csv",
            use_container_width=True
        )
    
    with progress_col:
        st.progress(1.0)
        st.caption("✅ Обработка завершена")
    
    st.markdown("---")
    st.subheader("📈 Графики")
    
    dashboard_data = analytics_use_case.get_dashboard_data()
    daily_good_bad = dashboard_data['daily_good_bad']
    today_dept = dashboard_data['today_department']
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("### 📅 По дням (последние 7 дней)")
        if daily_good_bad:
            daily_df = pd.DataFrame(daily_good_bad[-7:])
            if not daily_df.empty:
                daily_chart_df = daily_df.set_index('date')[['good', 'bad']]
                st.bar_chart(daily_chart_df)
            else:
                st.info("Нет данных")
        else:
            st.info("Нет данных")
    
    with chart_col2:
        st.markdown("### 🏪 По отделам (за сегодня)")
        if today_dept:
            today_df = pd.DataFrame(today_dept)
            if not today_df.empty:
                today_chart_df = today_df.set_index('department')[['good', 'bad']]
                st.bar_chart(today_chart_df, horizontal=True)
            else:
                st.info("Нет данных за сегодня")
        else:
            st.info("Нет данных за сегодня")
    
    if st.session_state.bad_crops_data and len(st.session_state.bad_crops_data) > 0:
        st.markdown("---")
        st.subheader(f"🗑️ Плохие ценники ({len(st.session_state.bad_crops_data)} шт.)")
        
        n_cols = 4
        n_rows = (len(st.session_state.bad_crops_data) + n_cols - 1) // n_cols
        
        for i, crop_data in enumerate(st.session_state.bad_crops_data):
            col_idx = i % n_cols
            if col_idx == 0:
                cols = st.columns(n_cols)
            
            with cols[col_idx]:
                caption = f"ID: {crop_data['track_id']}\nКачество: {crop_data['confidence']:.1f}%\nЦвет: {crop_data['color']}"
                if crop_data.get('product_name') and crop_data['product_name'] != 'Товар (OCR будет позже)':
                    caption += f"\n{crop_data['product_name']}"
                st.image(
                    f"data:image/png;base64,{crop_data['image_base64']}",
                    caption=caption,
                    use_container_width=True
                )