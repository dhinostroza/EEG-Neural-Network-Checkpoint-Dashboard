import streamlit as st
import pandas as pd
import os
import torch
import time
import altair as alt
from analyzer import CheckpointAnalyzer
from inference import get_model, load_checkpoint_weights, preprocess_spectrogram, detect_architecture, convert_edf_to_parquet



# ... imports remain the same ...

# ==============================================================================
# LOCALIZATION (i18n)
# ==============================================================================
TRANSLATIONS = {
    "page_title": {
        "en": "NSSR SHHS Checkpoint Dashboard",
        "es": "Panel de control de modelos NSSR SHHS"
    },
    "header_title": {
        "en": "Neural Network Checkpoint Dashboard",
        "es": "Panel de control de modelos (Checkpoints)"
    },
    "header_desc": {
        "en": "Monitor training results and identify top-performing models.",
        "es": "Monitorea los resultados del entrenamiento e identifica los modelos de mejor rendimiento. Los EEG fueron convertidos de formato .edf a .parquet, para acelerar la lectura. Para esto, se extrajo la lógica de procesamiento de señales del script original, utilizando `mne` para la lectura y re-muestreo a 64Hz, y `scipy` para la generación de espectrogramas (dimensiones 76x60, `nperseg=128`, log-scaling). Este proceso se ejecuta automáticamente en tiempo real al subir un archivo .edf. El análisis se hizo con el mejor modelo entrenado disponible y se procesó con numpy para optimizar la velocidad de estos resultados."
    },
    "settings": {
        "en": "Settings",
        "es": "Configuración"
    },
    "language": {
        "en": "Language",
        "es": "Idioma"
    },
    "refresh_btn": {
        "en": "Refresh / Scan New Files",
        "es": "Actualizar / escanear archivos"
    },
    "threshold_label": {
        "en": "Success Threshold (Val Loss)",
        "es": "Umbral de éxito (Val Loss)"
    },
    "tab_dashboard": {
        "en": "Model Dashboard",
        "es": "Tablero de modelos entrenados"
    },
    "tab_inference": {
        "en": "Inference Playground",
        "es": "Carga y resultados"
    },
    "tab_script": {
        "en": "Conversion Script",
        "es": "Script de conversión de .edf a .parquet"
    },
    "metric_total": {
        "en": "Total Checkpoints",
        "es": "Total de modelos (checkpoints)"
    },
    "metric_best_loss": {
        "en": "Best Validation Loss",
        "es": "Mejor pérdida (Val Loss)"
    },
    "metric_avg_params": {
        "en": "Avg Parameters",
        "es": "Promedio de parámetros (M = millones)"
    },
    "model_registry": {
        "en": "Model Registry",
        "es": "Modelos disponibles"
    },
    "no_checkpoints": {
        "en": "No checkpoints found. Click Refresh to scan.",
        "es": "No se encontraron checkpoints. Haga clic en Actualizar."
    },
    "inf_header": {
        "en": "Sleep Stage Inference",
        "es": "Inferencia de las etapas del sueño"
    },
    "inf_desc": {
        "en": "Upload a .parquet file to classify sleep stages using your trained models.",
        "es": "Cargue un archivo .parquet para clasificar las etapas de sueño usando el modelo predeterminado o vuelva a la ventana previa para seleccionar otro modelo."
    },
    "processing": {
        "en": "Processing...",
        "es": "Procesando..."
    },
    "upload_label": {
        "en": "Upload EEG Spectrograms (.parquet, .edf)",
        "es": "Cargar espectrogramas EEG (.parquet, .edf)"
    },
    "select_model": {
        "en": "Select a model. The best one is selected by default.",
        "es": "Seleccione un modelo. El mejor es el predeterminado."
    },
    "no_models_err": {
        "en": "No models available. Please scan checkpoints first.",
        "es": "No hay modelos disponibles. Escanee checkpoints primero."
    },
    "model_label": {
        "en": "Model",
        "es": "Modelo"
    },
    "arch": {
        "en": "Architecture",
        "es": "Arquitectura"
    },
    "val_loss": {
        "en": "Val Loss",
        "es": "Pérdida Val"
    },

    "epoch": {
        "en": "Epoch",
        "es": "Época"
    },
    "date_label": {
        "en": "Date",
        "es": "Fecha"
    },
    "time_label": {
        "en": "Time",
        "es": "Hora"
    },
    "files_label": {
        "en": "Files",
        "es": "Archivos"
    },
    "params_label": {
        "en": "Params",
        "es": "Parámetros"
    },
    "history_label": {
        "en": "Processed Files",
        "es": "Archivos procesados"
    },
    "select_model_label": {
        "en": "Select Model",
        "es": "Seleccionar el modelo"
    },
    "params_label": {
        "en": "Params",
        "es": "Parámetros"
    },
    "analyze_btn": {
        "en": "Analyze Sleep Stages",
        "es": "Analizar las etapas del sueño"
    },
    "loading_model": {
        "en": "Loading Model...",
        "es": "Cargando el modelo..."
    },
    "detected_arch": {
        "en": "Architecture detected",
        "es": "Arquitectura detectada"
    },
    "loading_data": {
        "en": "Loading Data...",
        "es": "Cargando los datos..."
    },
    "running_inf": {
        "en": "Running Inference on {} samples...",
        "es": "Ejecutando la inferencia en {} muestras..."
    },
    "analysis_complete": {
        "en": "Analysis Complete!",
        "es": "¡Análisis completo!"
    },
    "results_analysis": {
        "en": "Results Analysis",
        "es": "Análisis de los resultados"
    },
    "sleep_dist": {
        "en": "Sleep Stage Distribution",
        "es": "Distribución de las etapas del sueño"
    },
    "model_context": {
        "en": "Model Context",
        "es": "Contexto del modelo"
    },
    "high_reliability": {
        "en": "**High Reliability**: This model has successfully minimized error on the validation set.",
        "es": "**Alta confiabilidad**: este modelo ha minimizado exitosamente el error."
    },
    "mod_reliability": {
        "en": "**Moderate Reliability**: Performance is acceptable but may have some misclassifications.",
        "es": "**Confiabilidad moderada**: rendimiento aceptable pero puede tener errores."
    },
    "low_reliability": {
        "en": "**Low Reliability**: Inferences may be noisy or inaccurate.",
        "es": "**Baja confiabilidad**: las inferencias pueden ser inexactas."
    },
    "context_caption": {
        "en": "Context helps interpret if the distribution above is trustworthy.",
        "es": "El contexto ayuda a interpretar si la distribución es confiable."
    },
    "dist_table": {
        "en": "Distribution Table",
        "es": "Tabla de distribución"
    },
    "download_results": {
        "en": "Download Results",
        "es": "Descargar los resultados"
    },
# ... (Previous code)

    "download_csv": {
        "en": "Download Predictions CSV",
        "es": "Descargar el CSV de las predicciones"
    },
    "instant_success": {
        "en": "Loading results...",
        "es": "Cargando los resultados..."
    },
    "col_stage": {
        "en": "Sleep Stage",
        "es": "Etapa de sueño"
    },
    "col_count": {
        "en": "Count",
        "es": "Conteo"
    },
    "stage_wake": {
        "en": "Wake",
        "es": "Vigilia"
    }
}

# --- Language Helper ---
st.set_page_config(page_title="NSSR SHHS Dashboard", layout="wide")

def inject_custom_css():
    st.markdown("""
        <style>
        /* CSS Hack to translate File Uploader */
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] div div small {
            display: none;
        }
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] div div::after {
            content: "Límite 200MB por archivo • PARQUET, EDF";
            font-size: 0.8em;
        }
        
        /* Drag and Drop Text */
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] div div span {
             display: none;
        }
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] div div::before {
            content: "Arrastre y suelte archivos aquí";
            font-size: 1rem;
            display: block;
            margin-bottom: 5px;
        }

         /* Browse Button (This is harder to target perfectly without hiding interaction, be careful) */
         /* Usually better to leave 'Browse files' or accept it, but let's try to override if feasible. */
         /* The button is inside generated shadow DOM or complex sturcture sometimes. Skipping button to avoid breakage. */
         
         /* Force scrollbars to be visible (Mac auto-hide override) */
         /* Webkit browsers (Chrome, Safari) */
         ::-webkit-scrollbar {
            -webkit-appearance: none;
            width: 8px !important;
            height: 8px !important;
            display: block !important;
         }
         ::-webkit-scrollbar-thumb {
            border-radius: 4px;
            background-color: rgba(100, 100, 100, 0.5) !important; /* Visible grey */
            box-shadow: 0 0 1px rgba(255, 255, 255, .5);
         }
         ::-webkit-scrollbar-track {
             background-color: transparent;
         }
         
         /* Firefox */
         * {
            scrollbar-width: thin;
            scrollbar-color: rgba(100, 100, 100, 0.5) transparent;
         }
        </style>
    """, unsafe_allow_html=True)

# Call CSS immediately
inject_custom_css()

# Sidebar Language Selector (Placed First)
with st.sidebar:
    # Index 0 is English, Index 1 is Español. Defaulting to 1 (Español)
    lang_selection = st.radio("Language / Idioma", ["English", "Español"], index=1)
    LANG = 'en' if lang_selection == 'English' else 'es'

def t(key):
    return TRANSLATIONS.get(key, {}).get(LANG, key)

# --- Helpers ---
def retrieve_from_sql(filename):
    """
    Scans predictions.sql for the specific file block and parses it.
    Returns: List of predicted indices [0, 2, 1, ...]
    """
    sql_path = "predictions.sql"
    if not os.path.exists(sql_path):
        return None
        
    predictions = []
    
    # We look for the start marker
    start_marker = f"-- Data for {filename}"
    parsing = False
    
    # Simple file scan
    # For large files, grep is faster, but Python file reading is buffered and decent for <500MB
    # For 11k files, this file will be large, so we would ideally use a database.
    # For this demo, we can use a quick search or grep.
    try:
        # Optimization: Use grep to get the block line numbers or content if possible?
        # Or just read line by line. Let's try read line by line, if it's too slow we optimize.
        # Actually, let's use linux grep if available for speed.
        import subprocess
        
        # Grep extracting only lines matching the filename in the INSERT values
        # Format: ('SC4051', 'SC4051E.parquet', ...
        # We can just grep for the filename inside the file. 
        # But we need parsing.
        
        # Safer: Find the block start
        with open(sql_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if start_marker in line:
                    parsing = True
                    continue
                
                if parsing:
                    if line.startswith("-- Data for"): # Next block
                        break
                    
                    if "INSERT INTO" in line or "CREATE TABLE" in line:
                        continue
                        
                    # Parse values
                    # Line format: ('Client', 'File', 0, 'Stage', 0.99, 'Model'),
                    # We want the stage (index 3).
                    # 'Wake', 'N1', 'N2', 'N3', 'REM'
                    try:
                        # Extract the stage string
                        # split by comma, careful with quotes
                        parts = line.split(',')
                        if len(parts) >= 6:
                            stage_str = parts[3].strip().replace("'", "")
                            
                            # Map back to int for compatibility with app logic
                            stage_map_rev = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
                            if stage_str in stage_map_rev:
                                predictions.append(stage_map_rev[stage_str])
                    except:
                        continue
                        
    except Exception as e:
        print(f"Error reading SQL: {e}")
        return None
        
    except Exception as e:
        print(f"Error reading SQL: {e}")
        return None
        
    return predictions if predictions else None

def get_processed_files_list():
    """Reads processed_files.log to get list of files."""
    log_file = "processed_files.log"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            files = [line.strip() for line in f if line.strip()]
        return sorted(list(set(files)))
    return []

def save_results_to_sql(filename, predictions, confidence_scores, model_name, patient_id="UNKNOWN"):
    """Appends new results to SQL and Log."""
    sql_path = "predictions.sql"
    log_path = "processed_files.log"
    
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    
    try:
        # Append to SQL
        mode = 'a' if os.path.exists(sql_path) else 'w'
        with open(sql_path, mode) as f:
            if mode == 'w':
                 f.write("CREATE TABLE IF NOT EXISTS sleep_predictions ...;\n") # Simplified
            
            f.write(f"-- Data for {filename}\n")
            f.write("INSERT INTO sleep_predictions (patient_id, filename, epoch_index, predicted_stage, confidence, model_used) VALUES\n")
            
            values = []
            for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
                stage_label = stage_map.get(pred, "Unknown")
                val_str = f"('{patient_id}', '{filename}', {i}, '{stage_label}', {conf:.4f}, '{model_name}')"
                values.append(val_str)
            
            f.write(",\n".join(values))
            f.write(";\n")
            
        # Append to Log
        with open(log_path, "a") as f:
            f.write(filename + "\n")
            
        return True
    except Exception as e:
        print(f"Error saving to SQL: {e}")
        return False

# --- App Content ---

BASE_DIR = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/"
ANALYZER = CheckpointAnalyzer(BASE_DIR)

st.title(t("header_title"))
st.markdown(t("header_desc"))

# Initialize Session State
if 'df_models' not in st.session_state:
    st.session_state.df_models = pd.DataFrame()

# Sidebar Controls
st.sidebar.divider()
st.sidebar.header(t("settings"))
refresh_btn = st.sidebar.button(t("refresh_btn"))
loss_threshold = st.sidebar.slider(t("threshold_label"), 0.0, 1.0, 0.60, 0.01)

# Get processed files list (moved from sidebar)
processed_files = get_processed_files_list()
selected_history_files = [] # Initialize here for tab2

# Logic to load data
def load_data(force_refresh=False):
    if force_refresh:
        with st.spinner(t("refresh_btn")): # Using refresh text as spinner
            df = ANALYZER.scan()
            st.success("Scan complete!")
            return df
    else:
        df = pd.DataFrame(ANALYZER.cache.values())
        if df.empty:
            df = ANALYZER.scan()
        return df

if refresh_btn:
    st.session_state.df_models = load_data(force_refresh=True)
elif st.session_state.df_models.empty:
    st.session_state.df_models = load_data()

df = st.session_state.df_models

# --- TABS LAYOUT ---
# Auto-switch to inference tab if history is selected
if selected_history_files:
    # Use index 1 (Inference/Results) as default if selection exists
    # We can't robustly force-switch tabs in Streamlit without custom component hack or 
    # resetting the whole logic. 
    # BUT we can use the `st.tabs` indices if we manage state? No, tabs are UI containers.
    # We can just render the content. 
    # The user asks "shouldn't the page switch".
    # We can try to prioritize the tab default, but Streamlit tabs are client-side.
    # Better approach: If history selected, we might just SHOW the results? 
    # Or implies the user is ON tab 2? 
    # We can't change active tab easily.
    # However, we can use a session state variable for 'active_tab' if we used st.radio or similar.
    # With st.tabs, it's hard.
    # WORKAROUND: We can print a message or just rely on user being there.
    # BUT, if we can't switch, we can at least make sure it doesn't crash on Tab 1? 
    # No, the crash is in Tab 2 logic.
    pass

tab1, tab2, tab3 = st.tabs([t("tab_dashboard"), t("tab_inference"), t("tab_script")])

# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    # Display Metrics
    col1, col2, col3 = st.columns(3)
    if not df.empty:
        total_models = len(df)
        valid_models = df[df['val_loss'].notna()]
        best_loss = valid_models['val_loss'].min() if not valid_models.empty else 0
        avg_params = valid_models['params_m'].mean() if 'params_m' in valid_models.columns else 0

        col1.metric(t("metric_total"), total_models)
        col2.metric(t("metric_best_loss"), f"{best_loss:.4f}")
        col3.metric(t("metric_avg_params"), f"{avg_params:.1f} M")

        # Data Table
        st.subheader(t("model_registry"))
        
        # Styling
        def highlight_good_models(s):
            is_good = False
            try:
                if s['val_loss'] is not None and s['val_loss'] <= loss_threshold:
                    is_good = True
            except:
                pass
            return ['background-color: #d4edda' if is_good else '' for _ in s]

        # Clean up column view
        display_cols = ['date_modified', 'val_loss', 'model_architecture', 'params_m', 'epoch', 'size_mb', 'filename', 'status']
        display_cols = [c for c in display_cols if c in df.columns]
        
        st_df = df[display_cols].sort_values(by='date_modified', ascending=False)
        
        # Apply style
        styled_df = st_df.style.apply(highlight_good_models, axis=1).format({
            "val_loss": "{:.4f}",
            "size_mb": "{:.1f}",
            "params_m": "{:.2f}"
        })
        
        st.dataframe(styled_df, width="stretch", height=800)
    else:
        st.info(t("no_checkpoints"))

# ==============================================================================
# TAB 2: INFERENCE
# ==============================================================================
with tab2:
    # Create layout: 2 Main Columns
    # Col 1: Sidebar-like File List (Full height)
    # Col 2: Main Content (Uploader/Model on top, Results below)
    col_left_sidebar, col_main_content = st.columns([1.0, 9.0])
    
    selected_history_files = []
    
    # --- COLUMN 1: LEFT SIDEBAR (History) ---
    with col_left_sidebar:
        # Standardized Header
        st.markdown(f"**{t('history_label')}**")
        
        if processed_files:
             df_files = pd.DataFrame({"filename": processed_files})
             # Remove .parquet extension for display
             df_files["display_name"] = df_files["filename"].str.replace(".parquet", "", regex=False)
             
             # Increased height to act as a sidebar (700px)
             event = st.dataframe(
                df_files,
                column_config={
                    "display_name": st.column_config.TextColumn(t("files_label")),
                    "filename": None # Hide original filename
                },
                width="stretch",
                hide_index=True,
                height=700, 
                on_select="rerun",
                selection_mode="multi-row",
                key="history_list_main" # Unique key
            )
             if event.selection.rows:
                indices = event.selection.rows
                selected_history_files = df_files.iloc[indices]["filename"].tolist()
        else:
            st.info("-")

    # --- COLUMN 2: MAIN CONTENT ---
    with col_main_content:
        # Top Row: Uploader and Model Selector
        col_top_upload, col_top_model = st.columns([1.8, 7.2])
        
        # --- Top Left: Uploader ---
        with col_top_upload:
            st.markdown(f"**{t('upload_label')}**")
            uploaded_files = st.file_uploader(
                label=t("upload_label"),
                type=["parquet", "edf"],
                accept_multiple_files=True,
                key="uploader_main",
                label_visibility="collapsed"
            )
            
        # --- Top Right: Model Selection ---
        with col_top_model:
            # Select Model Logic (Default to Best)
            if df.empty:
                 st.warning(t("no_models_err"))
                 selected_model_name = None
            else:
                 valid_df = df[df['val_loss'].notna()].copy()
                 if not valid_df.empty:
                    # Identify best model for default
                    best_model_idx = valid_df['val_loss'].idxmin()
                    best_model_name = valid_df.loc[best_model_idx, 'filename']
                    
                    # Model Selector
                    model_options = valid_df['filename'].tolist()
                    # Find index of best model in the list
                    default_index = model_options.index(best_model_name) if best_model_name in model_options else 0
                    
                    # Standardized Header
                    st.markdown(f"**{t('select_model_label')}**")
                    
                    selected_model_name = st.selectbox(
                        label=t("select_model_label"),
                        options=model_options,
                        index=default_index,
                        key="model_selector",
                        label_visibility="collapsed"
                    )
                    
                    # Get the row for the SELECTED model
                    selected_model_row = df[df['filename'] == selected_model_name].iloc[0]
                    model_meta = selected_model_row # Update global meta reference
                    
                    # Restore detailed info
                    st.markdown(f"**Model: {selected_model_name}**")
                    
                    # Extract meta
                    lr = selected_model_row.get('lr', 'N/A')
                    wrs = selected_model_row.get('weighted_sampler', False)
                    weights = "WRS: On" if wrs else "WRS: Off"
                    workers = f"Workers: {selected_model_row.get('workers', 0)}"
                    n_files = selected_model_row.get('trained_on_files', 0)
                    
                    info_c1, info_c2, info_c3 = st.columns(3)
                    
                    with info_c1:
                        st.markdown(f"**{t('date_label')}:** {selected_model_row.get('date', 'N/A')}")
                        st.markdown(f"**{t('arch')}:** {selected_model_row.get('model_architecture', 'Unknown')}")
                    
                    with info_c2:
                        st.markdown(f"**{t('time_label')}:** {selected_model_row.get('time', 'N/A')}")
                        v_loss = selected_model_row.get('val_loss', 0)
                        st.markdown(f"**{t('val_loss')}:** {v_loss:.4f}" if isinstance(v_loss, (int, float)) else f"**{t('val_loss')}:** {v_loss}")
                        
                    with info_c3:
                         st.markdown(f"**{t('files_label')}:** {n_files}")
                         st.markdown(f"**{t('params_label')}:** {lr}, {weights}, {workers}")
                    
                    st.divider()
                 else:
                    selected_model_name = None

        # --- RESULTS AREA (Full Main Column Width) ---
        # Auto-trigger analysis
        # Only run if files are present AND model is selected
        if (uploaded_files or selected_history_files) and selected_model_name:
             # Determine source: History OR Upload
             files_to_process = []
             if selected_history_files:
                 class MockFile:
                     def __init__(self, name): 
                         self.name = name
                         self.from_history = True # Flag to identify
                 
                 files_to_process = [MockFile(name) for name in selected_history_files]
                 st.info(f"Viewing {len(files_to_process)} processed files from history.")
             else:
                 files_to_process = uploaded_files
     
             # Iterate over each file
             for i, uploaded_file in enumerate(files_to_process):

                 # Header instead of expander for cleaner look
                 st.divider()
                 st.markdown(f"### 📄 {uploaded_file.name}")
                 
                 # Always show processing initially
                 status_text = st.empty()
                 progress_bar = st.progress(0)
                 status_text.text(t("processing"))
                 
                 try:
                    # 1. Save uploaded file (Skip for History MockFiles)
                    is_history_file = hasattr(uploaded_file, 'from_history') and uploaded_file.from_history
                    
                    # Determine extension
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    temp_filename = f"temp_upload_{i}{file_ext}" # Use unique name
                    
                    if not is_history_file:
                        with open(temp_filename, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    else:
                        pass
                    
                    # EDF Conversion Logic (Only for new real uploads)
                    if file_ext == ".edf" and not is_history_file:
                        status_text.text(f"Converting {uploaded_file.name} to Parquet...")
                        
                        # Define output parquet path
                        parquet_filename = temp_filename.replace(".edf", ".parquet")
                        
                        # Run conversion
                        success, msg = convert_edf_to_parquet(temp_filename, parquet_filename)
                        
                        if not success:
                            st.error(f"Failed to convert {uploaded_file.name}: {msg}")
                            # Clean up temp edf
                            if os.path.exists(temp_filename): os.remove(temp_filename)
                            continue
                        
                        # Success
                        temp_filename = parquet_filename
                        file_ext = ".parquet" 
                        status_text.text(f"Conversion complete. Running inference...")

                    # Check cache for the ORIGINAL filename (edf or parquet)
                    cached_preds = retrieve_from_sql(uploaded_file.name)
                    
                    # Load Data immediately (only if parquet AND not history mock)
                    if file_ext == ".parquet" and not is_history_file:
                        input_df = pd.read_parquet(temp_filename)
                    else: 
                         # If history mock, we don't have file content, so empty DF.
                         # This relies on cached_preds being present.
                         input_df = pd.DataFrame() 
                    
                    predictions = []
                    
                    # ------------------------------------------------------------------
                    # FAST PATH: Check for Pre-computed Results (SQL)
                    # ------------------------------------------------------------------
                    if cached_preds:
                        # Instant load
                        time.sleep(0.5) 
                        progress_bar.progress(100)
                        predictions = cached_preds
                        
                        # Reconstruct input_df if missing (History case)
                        if input_df.empty:
                            input_df = pd.DataFrame(index=range(len(predictions)))
                            
                        status_text.empty()
                        progress_bar.empty()
                    else:
                        # ------------------------------------------------------------------
                        # NORMAL PATH: Run Inference (Real Uploads Only)
                        # ------------------------------------------------------------------
                        if file_ext == ".edf":
                            pass
                        
                        status_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        # 2. Get Model Path
                        model_path = os.path.join(BASE_DIR, selected_model_name)
                        
                        status_text.text(t("loading_model"))
                        progress_bar.progress(10)
                        
                        # Detection
                        arch = detect_architecture(model_path)
                        st.caption(f"{t('detected_arch')}: {arch}")
                        
                        model = get_model(model_name=arch, num_classes=5, pretrained=False)
                        model, _ = load_checkpoint_weights(model, model_path)
                        model.eval()
                        
                        status_text.text(t("loading_data"))
                        progress_bar.progress(30)
                        
                        status_text.text(t("running_inf").format(len(input_df)))
                        
                        # Inference Loop
                        with torch.no_grad():
                            total = len(input_df)
                            for j in range(total):
                                row = input_df.iloc[j]
                                cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage'] if c in input_df.columns]
                                if cols_to_drop:
                                    flat_data = row.drop(cols_to_drop).values
                                else:
                                    flat_data = row.values
                                
                                input_tensor = preprocess_spectrogram(flat_data)
                                input_tensor = input_tensor.unsqueeze(0)
                                
                                logits = model(input_tensor)
                                pred_idx = torch.argmax(logits, dim=1).item()
                                predictions.append(pred_idx)
                                
                                if j % max(1, int(total/10)) == 0:
                                    prog = 30 + int(60 * (j / total))
                                    progress_bar.progress(prog)
                        
                        # SAVE TO SQL
                        dummy_confs = [1.0] * len(predictions)
                        save_success = save_results_to_sql(uploaded_file.name, predictions, dummy_confs, selected_model_name)
                        if save_success:
                            st.toast(f"Saved {uploaded_file.name} to history!")
                            # Cleanup suffix logic here if needed
                                 
                        progress_bar.progress(100)
                        status_text.success(t("analysis_complete"))
                        status_text.empty()
                        progress_bar.empty()
                    
                    # Process Results
                    stage_map = {0: t("stage_wake"), 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
                    input_df['predicted_mid'] = predictions
                    input_df['predicted_label'] = input_df['predicted_mid'].map(stage_map)
                    
                    # Visualization
                    st.subheader(t("results_analysis"))
                    
                    counts = input_df['predicted_label'].value_counts().reset_index()
                    counts.columns = [t("col_stage"), t("col_count")]
                    
                    # Layout
                    col_viz, col_context = st.columns([2, 1])
                    
                    with col_viz:
                        st.markdown(f"#### {t('sleep_dist')}")
                        c_stage = t("col_stage")
                        c_count = t("col_count")
                        wake_lbl = t("stage_wake")
                        sort_order = [wake_lbl, 'N1', 'N2', 'N3', 'REM']
                        
                        chart = alt.Chart(counts).mark_bar().encode(
                            x=alt.X(c_stage, sort=sort_order, axis=alt.Axis(labelAngle=0)),
                            y=c_count,
                            color=alt.Color(c_stage, scale={"domain": sort_order, "range": ['#f5bf03', '#a7c7e7', '#779ecb', '#03396c', '#ff6b6b']}),
                            tooltip=[c_stage, c_count]
                        ).properties(height=350)
                        st.altair_chart(chart, width="stretch")

                    with col_context:
                        st.markdown(f"#### {t('model_context')}")
                        st.info(f"**{t('arch')}**: {model_meta.get('model_architecture', 'Unknown')}")
                        
                        # Loss Analysis
                        v_loss = model_meta.get('val_loss')
                        if pd.notna(v_loss):
                            st.write(f"**{t('val_loss')}**: `{v_loss:.4f}`")
                            if v_loss < 0.60:
                                st.success(t("high_reliability"))
                            elif v_loss < 0.80:
                                st.warning(t("mod_reliability"))
                            else:
                                st.error(t("low_reliability"))
                        else:
                             st.write(f"**{t('val_loss')}**: N/A")

                        st.write(f"**{t('epoch')}**: {model_meta.get('epoch', 'N/A')}")
                        st.write(f"**{t('params_label')}**: {model_meta.get('params_m', 0):.2f} M")
                        st.caption(t("context_caption"))
                    
                    col_res1, col_res2 = st.columns(2)
                    with col_res1:
                        st.write(t("dist_table"))
                        st.dataframe(counts, width="stretch")
                    
                    with col_res2:
                        st.write(t("download_results"))
                        csv = input_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=t("download_csv"),
                            data=csv,
                            file_name=f"predictions_{uploaded_file.name}.csv",
                            mime="text/csv",
                            key=f"download_{i}" # Unique key for each button
                        )

                    # ----------------------------------------------------------------------
                    # GROUND TRUTH COMPARISON
                    # ----------------------------------------------------------------------
                    gt_col = None
                    for potential_col in ['label', 'stage', 'sleep_stage', 'annotation']:
                        if potential_col in input_df.columns:
                            gt_col = potential_col
                            break
                    
                    if gt_col:
                        unique_vals = input_df[gt_col].unique()
                        valid_labels = [v for v in unique_vals if v >= 0]
                        
                        if valid_labels:
                            st.divider()
                            st.subheader("📊 Comparison with Ground Truth")
                            
                            comp_df = pd.DataFrame({
                                "Epoch": range(len(input_df)),
                                "Ground Truth (Index)": input_df[gt_col],
                                "Predicted (Index)": input_df['predicted_mid'],
                            })
                            
                            comp_df["Ground Truth"] = comp_df["Ground Truth (Index)"].map(stage_map).fillna("Unknown")
                            comp_df["Predicted"] = input_df['predicted_label']
                            comp_df["Match"] = comp_df["Ground Truth (Index)"] == comp_df["Predicted (Index)"]
                            
                            acc = comp_df["Match"].mean() * 100
                            st.markdown(f"**Agreement/Accuracy**: `{acc:.2f}%`")
                            
                            st.dataframe(comp_df[["Epoch", "Ground Truth", "Predicted", "Match"]], width="stretch", height=400)
                        else:
                            st.caption("No ground truth labels found (values are all -1/missing).")
                 except Exception as e:
                     st.error(f"Error: {e}")

# ==============================================================================
# TAB 3: CONVERTER SCRIPT
# ==============================================================================
with tab3:
    st.header(t("tab_script"))
    script_path = "pre_shhs_edf2parquet.py"
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            code_content = f.read()
            st.code(code_content, language="python")
    else:
        st.error(f"Script file not found: {script_path}")
