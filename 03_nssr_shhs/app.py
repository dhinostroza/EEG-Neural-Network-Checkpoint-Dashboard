import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import time
import shutil
import glob
import altair as alt
import xml.etree.ElementTree as ET
import importlib
import json # Added for notebook generation
import analyzer
importlib.reload(analyzer)
from analyzer import CheckpointAnalyzer
from inference import get_model, load_checkpoint_weights, preprocess_spectrogram, detect_architecture, convert_edf_to_parquet
from colab_data import COLAB_NOTEBOOKS



# ... imports remain the same ...

# ==============================================================================
# LOCALIZATION (i18n)
# ==============================================================================
TRANSLATIONS = {
    # Portada (Cover/Home)
    "tab_portada": {
        "en": "Cover",
        "es": "Portada"
    },
    "page_title": {
        "en": "NSSR SHHS Checkpoint Dashboard",
        "es": "Panel de control de modelos NSSR SHHS"
    },
    "header_title": {
        "en": "Neural Network Checkpoint Dashboard",
        "es": "Panel de control de modelos (Checkpoints)"
    },
    "header_desc": {
        "en": "Monitor training results and identify top-performing models. (Numpy vectorization is ~120k epochs/sec).",
        "es": "Monitorea los resultados del entrenamiento e identifica los modelos de mejor rendimiento. Los EEG fueron convertidos de formato .edf/.bdf a .parquet, para acelerar la lectura. Para esto, se extrajo la lógica de procesamiento de señales del script original, utilizando `mne` para la lectura y re-muestreo a 64Hz, y `scipy` para la generación de espectrogramas (dimensiones 76x60, `nperseg=128`, log-scaling). Este proceso se ejecuta automáticamente en tiempo real al subir un archivo .edf/.bdf. El análisis se hizo con el mejor modelo entrenado disponible y se procesó con numpy para optimizar la velocidad de estos resultados (la vectorización de Numpy es ~120k épocas/seg)."
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
        "en": "Refresh / Scan Models",
        "es": "Cargar los modelos nuevos"
    },
    "threshold_label": {
        "en": "Success Threshold (Val Loss)",
        "es": "Umbral de éxito (Val Loss)"
    },
    "loading": {
        "en": "Loading...",
        "es": "Cargando..."
    },
    "tab_env": {
        "en": "Environment Setup",
        "es": "Preparación del entorno"
    },
    "tab_scripts_general": {
        "en": "Scripts",
        "es": "Scripts"
    },
    "tab_dashboard": {
        "en": "Model Dashboard",
        "es": "Tablero de modelos entrenados"
    },
    "tab_inference": {
        "en": "Inference Playground",
        "es": "Carga y resultados"
    },
    "tab_batch": {
        "en": "Automated Batch Processing",
        "es": "Procesamiento por lotes"
    },
    "tab_script": {
        "en": "Conversion Script (.edf/.bdf to .parquet)",
        "es": "Script de conversión de .edf/.bdf a .parquet"
    },
    "tab_colab_reports": {
        "en": "Colab Notebooks Report",
        "es": "Reporte de Notebooks Colab"
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
        "es": "No se encontraron checkpoints. Dé clic en 'Actualizar'."
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
        "en": "Upload EEG recordings\n(format .parquet, .edf, .bdf, .xml)",
        "es": "Cargar los registros de EEG\n(formato .parquet, .edf, .bdf, .xml)"
    },
    "select_model": {
        "en": "Select a model. The best one is selected by default.",
        "es": "Seleccione un modelo. El mejor es el predeterminado."
    },
    "no_models_err": {
        "en": "No models available. Please scan checkpoints first.",
        "es": "No hay modelos disponibles. Escanee los checkpoints primero."
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
    "msg_loaded_cache": {
        "en": "Loaded cached results for {filename}",
        "es": "Se cargaron los resultados en caché para {filename}"
    },
    "history_label": {
        "en": "Sleep-EDFX & SHHS EEG",
        "es": "EEG de Sleep-EDFX y SHHS"
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
        "es": "**Confiabilidad moderada**: el rendimiento es aceptable pero puede tener errores."
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
        "es": "Etapa del sueño"
    },
    "col_count": {
        "en": "Count",
        "es": "Conteo"
    },
    "pred_counts": {
        "en": "Prediction Counts",
        "es": "Conteo de predicciones"
    },
    "pred_stage": {
        "en": "Predicted Stage",
        "es": "Etapa predicha"
    },
    "class_index": {
        "en": "Class Index",
        "es": "Índice de clase"
    },
    "overall_acc": {
        "en": "Overall Accuracy",
        "es": "Precisión global"
    },
    "pred_table_title": {
        "en": "Model Predictions",
        "es": "Predicciones del modelo"
    },
    "conf_matrix_title": {
        "en": "Confusion Matrix",
        "es": "Matriz de confusión"
    },
    "results_title": {
        "en": "Detailed Results",
        "es": "Resultados detallados"
    },
    "comp_title": {
        "en": "Ground Truth Comparison",
        "es": "Comparación con Ground Truth"
    },
    "stage_wake": {
        "en": "Wake",
        "es": "Vigilia"
    },
    "hypno_comp": {
        "en": "Hypnogram Comparison (First 300 Epochs)",
        "es": "Comparación del hipnograma (primeras 300 épocas)"
    },
    "batch_source_dir": {
        "en": "Source Directory (Full Path)",
        "es": "Directorio de origen (Ruta completa)"
    },
    "batch_scan_btn": {
         "en": "Scan for Pending Files",
         "es": "Buscar archivos pendientes"
    },
    "batch_start_btn": {
         "en": "Start Batch Processing",
         "es": "Iniciar procesamiento por lotes"
    },
    "batch_stop_btn": {
         "en": "Stop Processing",
         "es": "Detener procesamiento"
    },
    "batch_pending": {
         "en": "Pending Files",
         "es": "Archivos pendientes"
    },
    "batch_processed_count": {
         "en": "Already Processed",
         "es": "Ya procesados"
    },
    "batch_param_warning": {
         "en": "Please enter a valid directory.",
         "es": "Por favor ingrese un directorio válido."
    },
    "training_set_size_label": {
        "en": "Filter by Training Set Size (Folder)",
        "es": "Filtrar por tamaño del set de entrenamiento"
    },
    "all_option": {
        "en": "All",
        "es": "Todos"
    },
    "files_word": {
        "en": "files",
        "es": "registros"
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
            content: "Límite de 200MB por archivo .parquet, .edf, .bdf, .xml";
            font-size: 0.8em;
        }
        
        /* Force Uploader Height */
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
            min-height: 150px !important;
            padding: 30px !important; /* Add padding to prevent cramping */
            display: flex;
            flex-direction: column; /* Stack explicit elements */
            align-items: center; 
            justify-content: center;
        }

        /* Reduce Sidebar Top Padding */
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 1rem !important;
        }
        
        /* Hide DEFAULT Streamlit text (Drag & drop, Limit...) which are usually inside small/span */
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] small,
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] span {
             display: none !important;
        }
        
        /* Inject CUSTOM text */
        /* We use ::after on the main dropzone container to append our text */
        section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"]::after {
            content: "Arrastre y suelte los archivos aquí \\A Límite 200MB por archivo .parquet, .edf, .bdf, .xml";
            white-space: pre-wrap; /* Allows line break \A and wrapping */
            text-align: center;
            font-size: 1rem;
            color: inherit;
            margin-top: 10px;
            display: block;
        }
        
        /* HACK: Hide Pagination "Showing page 1 of 6" and enable scrollbar */
        div[data-testid="stFileUploader"] ul[data-testid="scroll-to-bottom-container"] + div, 
        div[data-testid="stFileUploaderPagination"] {
            display: none !important;
        }
        
        section[data-testid="stFileUploader"] section[data-testid="stFileUploader"] {
            overflow: hidden; /* Avoid double scrolling */
        }

        /* Make the file list scrollable */
        section[data-testid="stFileUploader"] ul {
            max-height: 200px;
            overflow-y: auto !important;
        }
         
         /* Force scrollbars to be visible (Mac auto-hide override) */
         /* Webkit browsers (Chrome, Safari) */
         /* Force scrollbars to be visible (Mac auto-hide override) */
         /* Webkit browsers (Chrome, Safari) */
         ::-webkit-scrollbar {
            -webkit-appearance: none;
            width: 8px !important;
            height: 8px !important;
         }
         
         ::-webkit-scrollbar-thumb {
            border-radius: 4px;
            background-color: rgba(100, 100, 100, 0.5) !important; /* Visible grey */
            box-shadow: 0 0 1px rgba(255, 255, 255, .5);
         }
         
         ::-webkit-scrollbar-track {
             background-color: transparent;
         }
         
         /* Remove the aggressive selector that was forcing scrollbars on titles/columns */
         /* The st.container(height=...) handles overflow automatically */
         
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
    # 1. Logo and Header (Request: Match specific screenshot layout)
    st.image("/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/images/globo.svg", width=100)
    
    st.markdown("""
    <div style="text-align: left; margin-top: 10px; margin-bottom: 20px;">
        <p style="font-weight: bold; font-size: 0.9em; margin-bottom: 5px;">
        Modelo de aprendizaje profundo para la detección de los patrones de onda cerebral N1 durante los periodos de sueño.
        </p>
        <p style="font-size: 0.85em; margin-bottom: 2px;">Universidad Politécnica Salesiana</p>
        <p style="font-size: 0.85em; margin-bottom: 2px;">Tesis de grado</p>
        <p style="font-size: 0.85em;">2026-01-08</p>
    </div>
    """, unsafe_allow_html=True)

    # 2. Language Selector
    # Index 0 is English, Index 1 is Español. Defaulting to 1 (Español)
    lang_selection = st.radio("Language / Idioma", ["English", "Español"], index=1)
    LANG = 'en' if lang_selection == 'English' else 'es'

def t(key):
    return TRANSLATIONS.get(key, {}).get(LANG, key)

# --- Globals for Local SHHS Paths ---
SHHS_XML_DIRS = [
    "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files/annotations-events-profusion/shhs1/",
    "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files/annotations-events-nsrr/shhs1/",
    "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files/annotations-events-profusion/shhs2/",
    "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files/annotations-events-nsrr/shhs2/"
]

def extract_gt_from_xml(xml_path):
    """
    Parses Sleep-EDF (Hypnogram) or NSRR (Profusion) XMLs.
    Returns a list of labels (0=W, 1=N1, etc.)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        labels = []
        
        # Map strings to our indices: 0=W, 1=N1, 2=N2, 3=N3, 4=REM
        stage_map_str = {
            'W': 0, '0': 0, 'Wake': 0,
            '1': 1, 'N1': 1,
            '2': 2, 'N2': 2,
            '3': 3, 'N3': 3,
            '4': 3, 'N4': 3, # Map Stage 4 to N3 (AASM)
            'R': 4, 'REM': 4, '5': 4
        }
        
        # 1. Sleep-EDF <SleepStage> with <Duration>
        # (Already handled in previous logic, but check if we have pure <SleepStages> list first or mixed)
        
        # 2. Profusion Epoch List <SleepStages><SleepStage>0</SleepStage>...
        # This is a flat list of epochs (usually 30s)
        sleep_stages_container = root.find("SleepStages")
        if sleep_stages_container is not None:
             # Just iterate children
             for stage_elem in sleep_stages_container.findall("SleepStage"):
                 val = stage_elem.text
                 if val in stage_map_str:
                     labels.append(stage_map_str[val])
                 else:
                     # Fallback or strict? 
                     # Usually just map simple ints
                     try:
                         v_int = int(val)
                         # Simple map: 0=W,1=1,2=2,3=3,4=3,5=R
                         if v_int == 5: labels.append(4)
                         elif v_int == 4: labels.append(3)
                         else: labels.append(v_int)
                     except:
                         labels.append(-1)
             return labels

        # 3. Sleep-EDF Format (SleepStage + Duration siblings)
        sleep_stages = root.findall(".//SleepStage")
        durations = root.findall(".//Duration")
        
        if sleep_stages and durations and len(sleep_stages) == len(durations):
            for stage_elem, dur_elem in zip(sleep_stages, durations):
                stage_str = stage_elem.text
                duration = float(dur_elem.text)
                n_epochs = int(duration / 30)
                label = stage_map_str.get(str(stage_str), -1) 
                labels.extend([label] * n_epochs)
            return labels

        # 4. NSRR / Profusion <ScoredEvent>
        scored_events = root.findall(".//ScoredEvent")
        if scored_events:
            for event in scored_events:
                concept_node = event.find("EventConcept")
                if concept_node is None: continue
                concept = concept_node.text
                
                dur_node = event.find("Duration")
                if dur_node is None: continue
                duration = float(dur_node.text)
                
                if "Sleep stage" in concept:
                    label = -1
                    if "W" in concept or "0" in concept: label = 0
                    elif "N1" in concept or "1" in concept: label = 1
                    elif "N2" in concept or "2" in concept: label = 2
                    elif "N3" in concept or "3" in concept: label = 3
                    elif "N4" in concept or "4" in concept: label = 3
                    elif "R" in concept or "5" in concept: label = 4
                    elif "Unscored" in concept or "?" in concept: label = -1
                    
                    n_epochs = int(duration / 30)
                    labels.extend([label] * n_epochs)
            return labels

        return []
    except Exception as e:
        print(f"XML Parsing Error: {e}")
        return []

# --- Logic in loop ---
# Find where the loop is processing files and inject the lookup.
# Since I cant see the loop line numbers in this small context window, 
# I will use replace_file_content on the function definition first, 
# then another call to modify the loop. This tool call is just defining the function and globals.


# --- Helpers ---
def retrieve_from_sql(filename):
    """
    Scans predictions.sql for the specific file block and parses it.
    Returns: List of predicted indices [0, 2, 1, ...]
    """
    sql_path = "predictions.sql"
    if not os.path.exists(sql_path):
        return None
        
def retrieve_from_sql(filename):
    """Retrieves predictions from SQL if they exist."""
    sql_path = "predictions.sql"
    if not os.path.exists(sql_path):
        return None
        
    predictions = []
    
    # Check original name OR processed name
    # If filename is "SC4001E.parquet", check that AND "SC4001E_processed.parquet"
    target_filenames = [filename]
    name, ext = os.path.splitext(filename)
    if not name.endswith("_processed"):
         target_filenames.append(f"{name}_processed{ext}")
    
    try:
        with open(sql_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            # Simple scan for matching blocks
            for target in target_filenames:
                start_marker = f"-- Data for {target}"
                matching_preds = []
                parsing = False
                
                for line in lines:
                    if start_marker in line:
                        parsing = True
                        continue
                    
                    if parsing:
                        if line.startswith("-- Data for"): 
                            # Hit another block
                            parsing = False
                            continue
                        
                        if "INSERT INTO" in line or "CREATE TABLE" in line:
                            continue
                            
                        # Parse values: ('Patient', 'Filename', Index, 'Stage', ...)
                        try:
                            if f"'{target}'" in line:
                                parts = line.split(',')
                                # Old schema: 6 items. New schema: 7 items (true_stage at end)
                                if len(parts) >= 6:
                                    stage_str = parts[3].strip().replace("'", "")
                                    # Map back
                                    stage_map_rev = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}
                                    idx = int(parts[2])
                                    
                                    pred_label = stage_map_rev.get(stage_str, -1)
                                    true_label = -1
                                    
                                    # If new schema with Ground Truth
                                    if len(parts) >= 7:
                                        gt_str = parts[6].strip().replace("'", "").replace(")", "").replace(";", "")
                                        if gt_str != "NULL" and gt_str in stage_map_rev:
                                            true_label = stage_map_rev[gt_str]
                                    
                                    if pred_label != -1:
                                        matching_preds.append((idx, pred_label, true_label))
                        except Exception as e_parse:
                             # print(f"Parse error: {e_parse}")
                             continue
                
                if matching_preds:
                    # Sort and return
                    matching_preds.sort(key=lambda x: x[0])
                    # Return tuple (preds_list, gt_list)
                    preds = [p[1] for p in matching_preds]
                    gts = [p[2] for p in matching_preds]
                    return preds, gts
                    
    except Exception as e:
        print(f"Error reading SQL: {e}")
        return None, None
        
    return None, None

def get_processed_files_list():
    """Reads processed_files.log to get list of files. Deduplicates by base name."""
    log_file = "processed_files.log"
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            raw_files = [line.strip() for line in f if line.strip()]
        
        # Deduplicate: shhs1-200001.parquet vs shhs1-200001_processed.parquet
        final_map = {}
        for fname in raw_files:
            # Base name strategy: remove extension, remove _processed
            base = fname.replace(".parquet", "").replace(".edf", "").replace("_processed", "")
            
            # Logic: If we haven't seen this base, take it.
            # If we HAVE seen it, prefer the one with "_processed"
            if base not in final_map:
                final_map[base] = fname
            else:
                current_stored = final_map[base]
                # If incoming has _processed and stored doesn't, upgrade
                if "_processed" in fname and "_processed" not in current_stored:
                    final_map[base] = fname
                    
        return sorted(list(final_map.values()))
    return []

def save_results_to_sql(filename, predictions, confidence_scores, model_name, patient_id="UNKNOWN", true_labels=None):
    """Appends new results to SQL and Log. Includes Ground Truth if available."""
    sql_path = "predictions.sql"
    log_path = "processed_files.log"
    
    stage_map = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
    
    try:
        # Append to SQL
        mode = 'a' if os.path.exists(sql_path) else 'w'
        with open(sql_path, mode) as f:
            if mode == 'w':
                 f.write("CREATE TABLE IF NOT EXISTS sleep_predictions (patient_id TEXT, filename TEXT, epoch_index INT, predicted_stage TEXT, confidence FLOAT, model_used TEXT, true_stage TEXT);\n")
            
            f.write(f"-- Data for {filename}\n")
            f.write("INSERT INTO sleep_predictions (patient_id, filename, epoch_index, predicted_stage, confidence, model_used, true_stage) VALUES\n")
            
            values = []
            for i, (pred, conf) in enumerate(zip(predictions, confidence_scores)):
                stage_label = stage_map.get(pred, "Unknown")
                
                # Handle Ground Truth
                true_stage_label = "NULL"
                if true_labels and i < len(true_labels):
                    tl = true_labels[i]
                    if tl != -1: # Valid label
                         true_stage_label = f"'{stage_map.get(tl, 'Unknown')}'"
                
                val_str = f"('{patient_id}', '{filename}', {i}, '{stage_label}', {conf:.4f}, '{model_name}', {true_stage_label})"
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

def create_notebook_json(code_content):
    """Wraps raw python code in a valid Jupyter Notebook JSON structure."""
    notebook_structure = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_content.splitlines()]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    return json.dumps(notebook_structure, indent=1)

# --- App Content ---

BASE_DIR = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/checkpoint_files/"
ANALYZER = CheckpointAnalyzer(BASE_DIR)

# Initialize Session State
if 'df_models' not in st.session_state:
    st.session_state.df_models = pd.DataFrame()

# Sidebar Controls
# Sidebar Controls
with st.sidebar:
    st.divider()
    st.header(t("settings"))
    refresh_btn = st.button(t("refresh_btn"))
    loss_threshold = st.slider(t("threshold_label"), 0.0, 1.0, 0.56, 0.01)

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
        # Always scan to ensure we check for file existence/renames on every load
        # scan() is optimized to skip processing if mtime hasn't changed.
        df = ANALYZER.scan()
        return df

if refresh_btn:
    st.session_state.df_models = load_data(force_refresh=True)
elif st.session_state.df_models.empty:
    st.session_state.df_models = load_data()

df = st.session_state.df_models
# DEBUG:
# st.write(f"DEBUG: Checkpoints Dataframe Shape: {df.shape}")
# if not df.empty:
#     st.write(f"Columns: {df.columns.tolist()}")
#     st.write(df.head())
# else:
#     st.write("DEBUG: DataFrame is empty from session_state")

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

# HACK: Manage Tab selection by just rendering them.
tab_portada, tab_env, tab_scripts, tab1, tab2, tab3, tab4, tab_glossary = st.tabs([
    t("tab_portada"),
    t("tab_env"), 
    t("tab_scripts_general"),
    t("tab_dashboard"), 
    t("tab_inference"), 
    t("tab_batch"), 
    t("tab_script"),
    "Glosario"
])

# ==============================================================================
# TAB -1: PORTADA (Thesis Info)
# ==============================================================================
with tab_portada:
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; margin-bottom: 2rem;">
        <h1 style="font-size: 2.2rem; margin-bottom: 0.5rem; line-height: 1.2;">Modelo de aprendizaje profundo para la detección de los patrones de onda cerebral N1 durante los periodos de sueño.</h1>
        <h3 style="font-weight: normal; margin-top: 0; color: #333;">Universidad Politécnica Salesiana</h3>
        <p style="font-style: italic; font-size: 1.1rem; color: #555; margin: 0.2rem 0;">Tesis de grado</p>
        <div style="display: flex; justify-content: center; gap: 60px; margin: 1.5rem 0;">
            <div style="text-align: center; font-weight: bold; font-size: 1.1rem;">
                Tutor: Ing. Jonathan Roberto Torres Castillo, PhD
            </div>
            <div style="text-align: center; font-weight: bold; font-size: 1.1rem;">
                Estudiante: Nelson Daniel Hinostroza Castaño, MD
            </div>
        </div>
        <p style="color: #666;">2026-01-08</p>
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# TAB 0: ENVIRONMENT PREPARATION (new)
# ==============================================================================
with tab_env:
    st.header(t("tab_env"))
    try:
        with open("environment_reports.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    except FileNotFoundError:
        st.error(f"Report file 'environment_reports.md' not found.")

# --- CONSTANTS ---
TIMELINE_HTML_EN = """
<div style="margin-bottom: 30px;">
<h3 style="border-bottom: 2px solid #3366ff; padding-bottom: 10px;">Pipeline Evolution Timeline</h3>
<style>
.timeline-item {
padding: 10px 0 10px 30px;
border-left: 2px solid #555;
position: relative;
}
.timeline-item::before {
content: '';
position: absolute;
left: -6px;
top: 15px;
width: 10px;
height: 10px;
background: #3366ff;
border-radius: 50%;
}
.timeline-date {
font-size: 0.85em;
color: #555;
font-family: monospace;
}
.timeline-title {
font-weight: bold;
font-size: 1.1em;
color: #222;
}
.timeline-desc {
font-size: 0.95em;
color: #444;
margin-top: 5px;
}
</style>
<div class="timeline-item">
<div class="timeline-date">May - August 15, 2025</div>
<div class="timeline-title">Legacy Matlab Baseline</div>
<div class="timeline-desc">
Original EEGSNet implementation in Matlab. Used raw .edf files and manual loops. 
<i>Slow I/O, no augmentation, limited scalability.</i>
</div>
</div>
<div class="timeline-item">
<div class="timeline-date">August 16-31, 2025</div>
<div class="timeline-title">Python / PyTorch Port</div>
<div class="timeline-desc">
Migrated to Python. Replicated EEGSNet in PyTorch. Implemented basic <code>SleepDataset</code> loading all data into RAM.
<i>Functional but hit memory limits with full SHHS dataset.</i>
</div>
</div>
<div class="timeline-item">
<div class="timeline-date">Sept 1-15, 2025</div>
<div class="timeline-title">Architecture Upgrade</div>
<div class="timeline-desc">
Adopted <b>ConvNeXt V2</b> (Tiny/Base), EfficientNetV2, ViT, and tested Transformers (Swin). 
Introduced Class Weights to handle imbalance (N1 rare).
</div>
</div>
<div class="timeline-item" style="border-left: 2px solid #3366ff;">
<div class="timeline-date" style="color: #3366ff;">Sept 20, 2025 (SMOTE)</div>
<div class="timeline-title" style="color: #3366ff;">High-Performance Streaming</div>
<div class="timeline-desc">
Implemented <b>ChunkedIterableDataset</b> with Consolidated .npy files. 
Enabled <code>bf16</code> mixed precision and automated checkpointing. SMOTE implementation.
<i>Can scale to infinite dataset size without RAM issues.</i>
</div>
</div>
</div>
"""

TIMELINE_HTML_ES = """
<div style="margin-bottom: 30px;">
<h3 style="border-bottom: 2px solid #3366ff; padding-bottom: 10px;">Cronología de la evolución del <i>pipeline</i></h3>
<style>
.timeline-item {
padding: 10px 0 10px 30px;
border-left: 2px solid #555;
position: relative;
}
.timeline-item::before {
content: '';
position: absolute;
left: -6px;
top: 15px;
width: 10px;
height: 10px;
background: #3366ff;
border-radius: 50%;
}
.timeline-date {
font-size: 0.85em;
color: #555;
font-family: monospace;
}
.timeline-title {
font-weight: bold;
font-size: 1.1em;
color: #222;
}
.timeline-desc {
font-size: 0.95em;
color: #444;
margin-top: 5px;
}
</style>
<div class="timeline-item">
<div class="timeline-date">Mayo a Agosto 15, 2025</div>
<div class="timeline-title">Línea base en <i>Matlab</i></div>
<div class="timeline-desc">
Implementación original de EEGSNet en <i>Matlab</i> con archivos .edf crudos y bucles manuales.
<i>E/S lenta, sin aumento de datos, escalabilidad limitada.</i>
</div>
</div>
<div class="timeline-item">
<div class="timeline-date">Agosto 16-31, 2025</div>
<div class="timeline-title">Migración a Python/PyTorch</div>
<div class="timeline-desc">
Se replicó EEGSNet en <i>PyTorch</i>. Se implementó un <code>SleepDataset</code> básico que cargaba todos los datos en RAM.
<i>Funcional, pero alcanzó límites de memoria con el dataset completo SHHS.</i>
</div>
</div>
<div class="timeline-item">
<div class="timeline-date">Sept 1-15, 2025</div>
<div class="timeline-title">Actualización de la arquitectura</div>
<div class="timeline-desc">
Adopción de <b>ConvNeXt V2</b> (Tiny/Base) y pruebas con EfficientNetV2, ViT y Transformers (Swin).
Introducción de pesos de clase para manejar el desequilibrio (N1 es infrecuente).
</div>
</div>
<div class="timeline-item" style="border-left: 2px solid #3366ff;">
<div class="timeline-date" style="color: #3366ff;">Sept 20, 2025 (SMOTE)</div>
<div class="timeline-title" style="color: #3366ff;"><i>Streaming</i> de alto rendimiento</div>
<div class="timeline-desc">
Implementación de <b>ChunkedIterableDataset</b> con archivos .npy consolidados.
Habilitación de precisión mixta <code>bf16</code> y <i>checkpointing</i> automatizado. Implementación de SMOTE.
<i>Puede escalar a un tamaño de dataset infinito sin problemas de RAM.</i>
</div>
</div>
</div>
"""

# ==============================================================================
# TAB 0.5: SCRIPTS (new)
# ==============================================================================
with tab_scripts:
    st.header(t("tab_scripts_general"))
    # Translation hardcoded here for simplicity or added to TRANSLATIONS dict
    info_text = "These scripts represent the high-performance pipeline used for training on the consolidated SHHS dataset (2025-09)."
    if LANG == 'es':
        info_text = "Estos scripts representan el <i>pipeline</i> de alto rendimiento utilizado para el entrenamiento en el <i>dataset</i> SHHS consolidado (2025-09)."
    
    st.markdown(f"<div style='background-color: #e6f3ff; color: #0066cc; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>{info_text}</div>", unsafe_allow_html=True)

    # --- TIMELINE VISUALIZATION ---
    timeline_html = TIMELINE_HTML_EN if LANG == 'en' else TIMELINE_HTML_ES
    st.markdown(timeline_html, unsafe_allow_html=True)

    # --- COLAB NOTEBOOKS REPORTS ---
    st.markdown("---")
    
    # Pre-defined Spanish translations for notebook descriptions
    NOTEBOOK_DESCRIPTIONS_ES = {
        '2025-09-01a ConvNeXt_tiny 500files GCS.ipynb': """
**Resumen:** Script inicial de entrenamiento que establece la línea base para el pipeline en la nube con ConvNeXt Tiny.

**Desglose del código:**
*   **1. SETUP:** Autentica el entorno de Colab con Google Cloud y monta Google Drive para guardar logs y checkpoints persistentes.
*   **2. DEPENDENCY INSTALLATION:** Instala librerías críticas (`pytorch-lightning`, `timm`, `gcsfs`) fijando versiones (`pandas==2.2.2`, `pyarrow==19.0.0`) para evitar conflictos de dependencias conocidos en Colab.
*   **3. IMPORTS:** Importa PyTorch y configura la precisión de multiplicación de matrices (`set_float32_matmul_precision`) para optimizar el uso de los Tensor Cores de la GPU.
*   **4. MODEL ARCHITECTURE:** Define `get_model`, que adapta la arquitectura ConvNeXt Tiny pre-entrenada para aceptar espectrogramas de 1 canal (en lugar de 3 RGB), promediando los pesos de la primera capa.
*   **5. LIGHTNING MODULE:** Define la clase `SleepStageClassifierLightning`, encapsulando la lógica de entrenamiento, validación, cálculo de pérdida (CrossEntropy) y configuración del optimizador (AdamW con Scheduler).
*   **6. DATASET:** Implementa `CombinedDataset`, que lee archivos Parquet directamente desde Google Cloud Storage (GCS), manejando la carga perezosa (lazy loading) y el caché en RAM.
*   **7. PERFORMANCE REPORT:** Función auxiliar para generar un reporte detallado (Precisión, Recall, F1 por clase) y la matriz de confusión al finalizar el entrenamiento.
*   **8. TRAINING EXECUTION:** Bloque principal que define hiperparámetros, divide el dataset (80/20), configura el `Trainer` de Lightning y ejecuta el ciclo de ajuste (`.fit()`).
""",
        '2025-09-01b ConvNext_base 1000files GCS.ipynb': """
**Resumen:** Expansión del experimento utilizando la arquitectura ConvNeXt Base y escalando a 1000 archivos.

**Desglose del código:**
*   **1. SETUP & 2. DEPENDENCIES:** Idéntico al script anterior, asegurando un entorno base consistente.
*   **3. IMPORTS:** Configuración estándar de PyTorch y Lightning.
*   **4. MODEL ARCHITECTURE:** Actualizado para instanciar `convnextv2_base`. Esta red es significativamente más profunda y ancha que la version Tiny, requiriendo más VRAM pero ofreciendo mayor capacidad de aprendizaje.
*   **6. DATASET:** Ajustado para manejar un volumen mayor de datos, poniendo a prueba la latencia de lectura desde GCS.
*   **8. EXECUTION:** Se incrementa el número de archivos a 1000 y se ajustan los pesos de clase (`CLASS_WEIGHTS`) para penalizar más los errores en la clase N1.
""",
        '2025-09-02 ConvNextv2_base cosine_annealing_lr-2e-5_epochs=40 2000files_w2 GCS.ipynb': """
**Resumen:** Implementación avanzada con programación de tasa de aprendizaje Cosine Annealing en un dataset de 2000 archivos.

**Desglose del código:**
*   **5. LIGHTNING MODULE (Update):** Se modifica `configure_optimizers` para usar `CosineAnnealingLR` en lugar de `ReduceLROnPlateau`. Esto permite que la tasa de aprendizaje decaiga suavemente siguiendo una función coseno, lo que a menudo ayuda a converger a mejores mínimos en entrenamientos largos (40 épocas).
*   **8. EXECUTION:** Se aumenta la escala a 2000 archivos. Se observa el uso de `NUM_WORKERS=2` para intentar paralelizar la carga de datos, aunque esto presenta desafíos de memoria en Colab que scripts futuros resolverán.
""",
        '2025-09-04 Performance Plot': "Visualización de Rendimiento. Reporte gráfico externo que muestra las curvas de aprendizaje. No contiene código ejecutable de entrenamiento, sino que carga los logs generados por los scripts anteriores para visualizar la convergencia y detectar overfitting.",
        '2025-09-06 ConvNeXTv2 base_epochs=40_1000files_lr-2e-5_gpu.ipynb': """
**Resumen:** Script de optimización crítica centrado en la aceleración por hardware (GPU Normalization).

**Desglose del código:**
*   **5. LIGHTNING MODULE (Optimización):** Se introduce el método `normalize_on_gpu` dentro del módulo Lightning.
    *   *Antes:* La normalización (restar media, dividir por std) se hacía en la CPU en `__getitem__`, cuello de botella.
    *   *Ahora:* Los datos crudos (int/float) pasan a la GPU, y la normalización masiva ocurre allí usando tensores de PyTorch. Esto acelera drásticamente el tiempo por época.
*   **6. DATASET:** Se simplifica el método `__getitem__` para eliminar operaciones NumPy costosas, delegando todo el trabajo pesado a la GPU.
""",
        '2025-09-10 Manifest multiple scripts CONVNEXT 2000files Training.ipynb': """
**Resumen:** Introducción de 'Manifests' y Entornos Virtuales aislados para reproducibilidad robusta.

**Desglose del código:**
*   **1. AUTHENTICATION:** Autenticación inicial para obtener credenciales que se pasarán al sub-entorno.
*   **2. & 3. VIRTUALENV:** Creación programática de un entorno virtual (`virtualenv manifest_env`). Esto aisla completamente las dependencias del script de las de Colab, evitando conflictos irresolubles.
*   **Step 5. PYTHON LOGIC:** El script de Python que genera el manifiesto se escribe como una cadena de texto (string) y se guarda en disco.
    *   **Lógica:** Escanea todos los buckets, valida cada archivo Parquet abriéndolo, cuenta las épocas válidas y escribe una entrada en `shhs_dataset_manifest.csv`.
    *   **Checkpointing:** Guarda el progreso cada 200 archivos para permitir reanudar si Colab se desconecta.
""",
        '2025-09-11 Consolidate spectrograms_labels_dataset into large files.ipynb': """
**Resumen:** Experimento fallido (LEGADO) de consolidación de datos.

**Desglose del código:**
*   **Intento:** Leer miles de archivos pequeños y concatenarlos en dos array gigantes de NumPy (`.npy`), guardándolos en disco.
*   **Problema:** Se demostró que cargar archivos de >10GB en memoria causaba crashes, y el formato PyArrow tuvo problemas de compatibilidad al intentar leer estos bloques masivos posteriormente. Se mantiene como documentación de "qué no hacer".
""",
        '2025-09-14 Training script manifest and sampler.ipynb': """
**Resumen:** Solución técnica al desequilibrio de clases usando `WeightedRandomSampler`.

**Desglose del código:**
*   **6. DATASET (Update):** Añade el método `get_labels_for_sampler` que escanea eficientemente solo la columna de etiquetas de todo el dataset antes del entrenamiento.
*   **8. EXECUTION:**
    *   Calcula la distribución de clases (e.g., N2 es mucho más frecuente que N1).
    *   Crea un `WeightedRandomSampler` de PyTorch. Este objeto se pasa al `DataLoader`.
    *   **Efecto:** En cada batch, el sampler fuerza la inclusión de ejemplos raros (N1) con mayor probabilidad, equilibrando artificialmente lo que ve el modelo y mejorando el aprendizaje de clases difíciles.
""",
        '2025-09-20 Consolidate_Dataset_Small Chunks VENV multiple files OK.ipynb': """
**Resumen:** Solución definitiva de ingeniería de datos (Chunking).

**Desglose del código:**
*   **Estrategia:** En lugar de 1 archivo gigante (fallido) o 5000 pequeños (lento), se consolidan grupos de 50 archivos originales en 1 archivo `.npy` consolidado ("Chunk").
*   **Implementación:** Usa un entorno virtual seguro. Itera sobre el manifiesto, agrupa rutas de 50 en 50, lee los datos y guarda un archivo `chunk_X.npy`.
*   **Resultado:** Reduce las operaciones de I/O en un factor de 50x sin exceder la memoria RAM. Es el formato usado para el entrenamiento final.
""",
        '2025-09-22 Google Drive to GCS Migration Script': """
**Resumen:** Herramienta de infraestructura para migración de datos.

**Desglose del código:**
*   **Utilidad:** Usa comandos `gsutil -m cp -r` (multithreaded copy) para mover gigabytes de datos pre-procesados desde Drive a Buckets de Google Cloud Storage.
*   **Importancia:** GCS ofrece un ancho de banda mucho mayor que Drive, necesario para alimentar GPUs A100 sin pausas.
""",
        '2025-10-04 List ACTIVE DATASET FILES.ipynb': "Utilidad de auditoría. Script simple que lista el contenido del directorio de entrenamiento final para generar un registro de auditoría de los archivos exactos utilizados en el modelo de producción."
    }
    
    if LANG == 'es':
        st.subheader("Reporte detallado de los Notebooks (2025-09)")
        st.markdown("""
        A continuación se presentan los scripts originales utilizados en Google Colab, en orden cronológico. 
        Cada bloque contiene el código completo extraído del notebook.
        """)
    else:
        st.subheader("Detailed Notebooks Report (2025-09)")
        st.markdown("""
        Below are the original scripts used in Google Colab, in chronological order. 
        Each block contains the full code extracted from the notebook.
        """)

    sorted_filenames = sorted(COLAB_NOTEBOOKS.keys())
    
    for filename in sorted_filenames:
        info = COLAB_NOTEBOOKS[filename]
        title = info['title']
        description = info['description']
        
        code = info.get('code')
        image = info.get('image')
        
        # --- ROBUST TRANSLATION LOOKUP ---
        # Debugging: Show the exact key we are trying to match
        # st.text(f"DEBUG: Processing key '{filename}'") 
        
        if LANG == 'es':
            # 1. Try exact match
            if filename in NOTEBOOK_DESCRIPTIONS_ES:
                description = NOTEBOOK_DESCRIPTIONS_ES[filename]
            else:
                # 2. Try normalized matching (ignore .ipynb and whitespace)
                norm_filename = filename.replace('.ipynb', '').strip()
                found_match = False
                for es_key, es_desc in NOTEBOOK_DESCRIPTIONS_ES.items():
                    norm_es_key = es_key.replace('.ipynb', '').strip()
                    if norm_filename == norm_es_key:
                        description = es_desc
                        found_match = True
                        break
                
                # if not found_match:
                #     st.warning(f"No translation found for: {filename}")

        # Determine icon based on type
        
        # Determine icon based on type
        icon = "📊" if "Plot" in title else "📓"
        
        # Localize label
        desc_label = "**Descripción:**" if LANG == 'es' else "**Description:**"
        
        with st.expander(f"{icon} {title}", expanded=False):
            st.markdown(f"{desc_label} {description}")
            
            if image:
                st.image(image, caption=title)
            
            if code:
                # Download button
                # User requested to leave extension as .ipynb
                btn_label = f"Descargar {filename}" if LANG == 'es' else f"Download {filename}"
                
                # Convert raw code string to valid .ipynb JSON
                notebook_json = create_notebook_json(code)
                
                st.download_button(
                    label=btn_label,
                    data=notebook_json,
                    file_name=filename,
                    mime='application/x-ipynb+json', 
                    key=f"dl_{filename}"
                )
                
                # Scrollable code container (600px height ~ 25 lines + scroll)
                try:
                    with st.container(height=600):
                        st.code(code, language='python')
                except TypeError:
                    st.warning("⚠️ Scrollable container not supported. Displaying full code.")
                    st.code(code, language='python')
# ==============================================================================
# TAB 1: DASHBOARD
# ==============================================================================
with tab1:
    st.title(t("header_title"))
    st.markdown(t("header_desc"))
    
    # 1. Group Selector
    # Get unique groups
    if not df.empty and 'group' in df.columns:
        # Custom sort: Extract number from string (e.g., '2000 files' -> 2000)
        def get_sort_key(s):
            # Handle '20-40 files' or '100 files'
            import re
            match = re.search(r'\d+', str(s))
            return int(match.group()) if match else 0
            
        raw_groups = df['group'].unique().tolist()
        # Sort descending by number
        groups = sorted(raw_groups, key=get_sort_key, reverse=True)
        
        # Translation helper for the dropdown
        def format_group_name(group_name):
            if group_name == "All":
                return t("all_option")
            
            # Translate 'files' -> 'registros' if Spanish
            if LANG == 'es':
                return group_name.replace("files", t("files_word"))
            return group_name
        
        # Default to '2000 files' if exists
        idx = groups.index('2000 files') if '2000 files' in groups else 0
        
        selected_option = st.selectbox(
            t("training_set_size_label"), 
            ["All"] + groups, 
            index=idx + 1 if '2000 files' in groups else 0,
            format_func=lambda x: format_group_name(x) if x != "All" else t("all_option")
        )
        
        # Map back correctly (if "All" is selected)
        selected_group = selected_option
        
        # Filter DataFrame
        if selected_group != "All":
            df_display = df[df['group'] == selected_group].copy()
        else:
            df_display = df.copy()
            
        # --- DETAILED REPORTS LOGIC ---
        
        REPORTS = {
            "20-40 files": {
                "en": """
            ### Analysis: Small Dataset Baseline (20-40 Files)
            
            This folder represents the **baseline** experiments using a very small subset of data (20 to 40 recordings).
            
                *   **Performance**: The Validation Loss is very high (**> 1.30**).
                *   **Interpretation**: 
                    *   The model is **underfitting**. There is simply not enough data for the deep learning network to generalize the complex patterns of sleep stages (N1, N2, REM, etc.).
                    *   This serves as a "sanity check" to ensure the training pipeline acts correctly before determining if more data improves performance.
                
                #### **Model Naming & Learning Rates**
                *   **What is "lightning"?**: The prefix `lightning-` references **PyTorch Lightning**, the framework managing the training loop and checkpointing.
                *   **Architecture: ConvNext V2 Tiny**: 
                    *   **ConvNext V2** is a modern Convolutional Neural Network (CNN) designed by Meta to compete with Transformers. It uses "pure" convolution layers but modernized with Transformer design principles (e.g., larger kernels, inverted bottlenecks).
                    *   **"Tiny"**: This is the smallest version of the family. It has fewer parameters, making it **faster** to train and **less prone to overfitting** on smaller datasets (like this one) compared to "Base" or "Large" models.
                *   **Learning Rates (LR)**:
                    *   `lr-1e-05` (0.00001): **Conservative**. Very small steps. Safe but slow.
                    *   `lr-0_0001` (0.0001): **Standard**. A balanced approach for this architecture.
                    *   `lr-0_0005` (0.0005): **Aggressive**. Faster learning but higher risk of instability.
                
                #### **Training Progress: Steps vs. Epochs**
                *   **Epochs**: One "Epoch" means the model has seen **every single file** in the dataset once. 
                    *   Files like `epoch=02` or `epoch=04` are early snapshots (saved when validation improved).
                    *   The file `epoch=19` represents a model trained for much longer (20 full cycles), allowing it to learn more fine-grained details.
                *   **Steps (e.g., `step=380`)**: A "Step" is one gradient update using a single batch of data. 
                    *   `step=380` indicates that over the course of 19 epochs, the model updated its weights 380 times. 
                """,
            "es": """
                ### Análisis: línea base con datos limitados (20-40 registros)
                
                Esta carpeta representa los experimentos de **línea base** utilizando un subconjunto muy pequeño de datos (20 a 40 grabaciones).
                
                *   **Rendimiento**: La pérdida de validación (Val Loss) es muy alta (**> 1.30**).
                *   **Interpretación**: 
                    *   El modelo presenta **subajuste (underfitting)**. Simplemente no hay suficientes datos para que la red neuronal generalice los patrones complejos de las etapas del sueño (N1, N2, REM, etc.).
                    *   Esto sirve como una "prueba de cordura" para asegurar que el pipeline de entrenamiento funcione correctamente antes de determinar si más datos mejoran el rendimiento.
                
                #### **Nombres de modelos y tasas de aprendizaje**
                *   **¿Qué significa "lightning"?**: El prefijo `lightning-` se refiere a **PyTorch Lightning**, el framework que gestiona el bucle de entrenamiento y los checkpoints.
                *   **Arquitectura: ConvNext V2 Tiny**:
                    *   **ConvNext V2** es una Red Neuronal Convolucional (CNN) moderna diseñada por Meta para competir con los Transformers. Utiliza capas de convolución "puras" pero modernizadas con principios de diseño de Transformers.
                    *   **"Tiny"**: Es la versión más pequeña de la familia. Tiene menos parámetros, lo que la hace **más rápida** de entrenar y **menos propensa al sobreajuste** en conjuntos de datos pequeños (como este) en comparación con los modelos "Base" o "Large".
                *   **Tasas de Aprendizaje (LR)**:
                    *   `lr-1e-05` (0.00001): **Conservador**. Pasos muy pequeños. Seguro pero lento.
                    *   `lr-0_0001` (0.0001): **Estándar**. Un enfoque equilibrado para esta arquitectura.
                    *   `lr-0_0005` (0.0005): **Agresivo**. Aprendizaje más rápido pero mayor riesgo de inestabilidad.
                
                #### **Progreso del entrenamiento: Steps vs. Epochs**
                *   **Epochs (Épocas)**: Una "Época" significa que el modelo ha visto **cada uno de los archivos** del dataset una vez.
                    *   Archivos como `epoch=02` o `epoch=04` son instantáneas tempranas.
                    *   El archivo `epoch=19` representa un modelo entrenado por mucho más tiempo (20 ciclos completos), permitiéndole aprender detalles más finos.
                *   **Steps (Pasos) (ej. `step=380`)**: Un "Paso" es una actualización de pesos usando un solo lote (batch) de datos.
                    *   `step=380` indica que a lo largo de 19 épocas, el modelo actualizó sus pesos 380 veces.
                """
        },
            
            "50 files": {
                "en": """
            ### Analysis: Early Learning (50 Files)
            
            *   **Best Model**: `...epoch=05-val_loss=0.6241.ckpt`
            *   **Performance**: Metric improved significantly from the 20-40 file baseline (Loss dropped from ~1.3 to **0.62**).
            *   **Insight**: This drop indicates that **50 files** provide the minimum viable signal for the ConvNext architecture to start learning meaningful features, although it is still far from the optimal performance seen in larger sets.
            """,
                "es": """
            ### Análisis: aprendizaje temprano (50 registros)
            
            *   **Mejor modelo**: `...epoch=05-val_loss=0.6241.ckpt`
            *   **Rendimiento**: La métrica mejoró significativamente respecto a la línea base (la pérdida cayó de ~1.3 a **0.62**).
            *   **Insight**: Esta caída indica que **50 registros** proporcionan la señal mínima viable para que la arquitectura ConvNext comience a aprender características significativas, aunque todavía está lejos del rendimiento óptimo observado en conjuntos más grandes.
            """
            },
            
            "100 files": {
                "en": """
            ### Comparative Analysis: Architecture Search (100 Files)
            
            This folder contains an experiment comparing different neural network architectures on the same small dataset.
            
            | Architecture | Val Loss (Lower is Better) | Insight |
            | :--- | :--- | :--- |
            | **ConvNext Tiny** | **0.5867** | **Winner**. Smaller models often generalize better on small datasets (less overfitting). |
            | ConvNext Base | 0.6011 | Slightly worse. The extra parameters didn't help yet. |
            | ViT Base | 0.6536 | Vision Transformers typically need *massive* datasets to beat CNNs. |
            | EfficientNet B0 | 0.7169 | Performed worst in this spectrogram task. |
            
            **Conclusion**: For smaller subsets, efficient/tiny models ("Tiny") are superior to massive ones.
            """,
                "es": """
            ### Análisis comparativo: búsqueda de arquitectura (100 registros)
            
            Esta carpeta contiene un experimento que compara diferentes arquitecturas de redes neuronales en el mismo conjunto de datos pequeño.
            
            | Arquitectura | Val Loss (Menor es mejor) | Insight |
            | :--- | :--- | :--- |
            | **ConvNext Tiny** | **0.5867** | **Ganador**. Los modelos más pequeños suelen generalizar mejor en conjuntos de datos pequeños (menor sobreajuste). |
            | ConvNext Base | 0.6011 | Ligeramente peor. Los parámetros extra no ayudaron todavía. |
            | ViT Base | 0.6536 | Los Vision Transformers típicamente necesitan conjuntos de datos *masivos* para superar a las CNNs. |
            | EfficientNet B0 | 0.7169 | Tuvo el peor desempeño en esta tarea de espectrogramas. |
            
            **Conclusión**: Para subconjuntos más pequeños, los modelos eficientes ("Tiny") son superiores a los masivos.
            """
            },
            
            "200 files": {
                "en": """
            ### Analysis: 200 Files
            
            *   **Best Model**: `...convnext_tiny...val_loss=0.6182.ckpt`
            *   **Observation**: The loss (**0.6182**) is actually slightly *higher* than the best 100-file run (0.58).
            *   **Why?** Deep Learning training is stochastic. Sometimes a specific run (random initialization) or a specific subset of 200 patients (maybe including harder cases) can lead to slightly higher variance. It suggests that hyperparameters might need tuning as data size grows.
            """,
                "es": """
            ### Análisis: 200 registros
            
            *   **Mejor modelo**: `...convnext_tiny...val_loss=0.6182.ckpt`
            *   **Observación**: la pérdida (**0.6182**) es en realidad ligeramente *mayor* que la mejor ejecución de 100 archivos (0.58).
            *   **¿Por qué?** El entrenamiento de aprendizaje profundo es estocástico. A veces una ejecución específica (inicialización aleatoria) o un subconjunto específico de 200 pacientes (tal vez incluyendo casos más difíciles) puede llevar a una varianza ligeramente mayor. Sugiere que los hiperparámetros podrían necesitar ajuste a medida que crece el tamaño de los datos.
            """
            },
            
            "500 files": {
                "en": """
            ### Analysis: 500 Files (Hyperparameter Tuning)
            
            *   **Best Model**: `...convnext_tiny...val_loss=0.6432.ckpt`
            *   **Observation**: Performance remains around ~0.64. 
            *   **Key Takeaway**: Simply adding more data (100 -> 500) does not guarantee linear improvement. At this stage, the bottleneck might shift from "Data Quantity" to "Hyperparameters" (Learning Rate, Regularization). The model might be getting "confused" by the noise in larger datasets without adjusted augmentation/regularization.
            """,
                "es": """
            ### Análisis: 500 registros (ajuste de hiperparámetros)
            
            *   **Mejor modelo**: `...convnext_tiny...val_loss=0.6432.ckpt`
            *   **Observación**: el rendimiento se mantiene alrededor de ~0.64.
            *   **Conclusión clave**: Simplemente agregar más datos (100 -> 500) no garantiza una mejora lineal. En esta etapa, el cuello de botella podría cambiar de "Cantidad de datos" a "Hiperparámetros" (Tasa de aprendizaje, Regularización). El modelo podría estar "confundiéndose" por el ruido en conjuntos de datos más grandes si no se ajusta el aumento de datos/regularización.
            """
            },
            
            "1000 files": {
                "en": """
            ### Analysis: Architecture & Learning Rate (1000 Files)
            
            This folder highlights two critical experiments: fitting a **Vision Transformer (Swin)** and tuning the **Learning Rate**.
            
            #### **1. CNN vs. Transformer**
            
            | Model | Arch | Val Loss | Insight |
            | :--- | :--- | :--- | :--- |
            | **Model A** | **ConvNext Base** | **0.5837** | **Winner**. CNNs have "inductive bias" (they assume local patterns matter), which works great for spectrograms. |
            | **Model C** | Swin Base | 0.5951 | **Slightly Worse**. Swin is a Transformer. Transformers usually need *massive* data (millions of images) to beat CNNs because they have to "learn how to see" from scratch. 1000 files isn't enough yet. |
            
            #### **2. The Importance of Learning Rate**
            
            Comparing two identical ConvNext models:
            
            | Model | Learning Rate | Val Loss | Status |
            | :--- | :--- | :--- | :--- |
            | **Model A** | `2e-05` (0.00002) | **0.5837** | **Success**. |
            | **Model B** | `5e-06` (0.000005) | 0.8167 | **Failed**. The LR was too small, causing the model to learn too slow and get stuck. |
            """,
                "es": """
            ### Análisis: arquitectura y tasa de aprendizaje (1000 registros)
            
            Esta carpeta destaca dos experimentos críticos: probar un **Vision Transformer (Swin)** y ajustar la **Tasa de aprendizaje**.
            
            #### **1. CNN vs. Transformer**
            
            | Modelo | Arq | Val Loss | Insight |
            | :--- | :--- | :--- | :--- |
            | **Modelo A** | **ConvNext Base** | **0.5837** | **Ganador**. Las CNN tienen "sesgo inductivo" (asumen que los patrones locales importan), lo cual funciona muy bien para espectrogramas. |
            | **Modelo C** | Swin Base | 0.5951 | **Ligeramente peor**. Swin es un Transformer. Los Transformers suelen necesitar datos *masivos* (millones de imágenes) para superar a las CNN porque tienen que "aprender a ver" desde cero. 1000 registros no son suficientes todavía. |
            
            #### **2. La importancia de la tasa de aprendizaje (Learning Rate)**
            
            Comparando dos modelos ConvNext idénticos:
            
            | Modelo | Learning Rate | Val Loss | Estado |
            | :--- | :--- | :--- | :--- |
            | **Modelo A** | `2e-05` (0.00002) | **0.5837** | **Éxito**. |
            | **Modelo B** | `5e-06` (0.000005) | 0.8167 | **Fallido**. El LR era demasiado pequeño, causando que el modelo aprendiera demasiado lento y se estancara. |
            """
            },
            
            "2000 files": {
                "en": """
            ### Comparative Analysis: 2000 Files Training Set
            
            This folder contains models trained on a subset of **2000** files from the SHHS dataset.
            
            #### **1. Best Model vs. Augmented Model**
            We compare the two primary candidates in this folder:
            
            | Feature | **Model A (Sept 4)** | **Model B (Sept 9)** |
            | :--- | :--- | :--- |
            | **Filename** | `...workers2.ckpt` | `...Augmented_cwN1-6.5.ckpt` |
            | **Validation Loss** | **0.5533** (Lower is "Better") | 0.6349 |
            | **Class Weights** | cwN1-8.0 | **cwN1-6.5** |
            | **Augmentation** | Standard | **Heavy Augmentation** |
            
            #### **2. Educational Concepts**
            
            **What is cwN1? (Class Weight for N1)**
            *   **The Problem**: In sleep staging data, Stage N1 is very rare (typically ~5% of epochs), while N2 and Wake are abundant. A standard model can achieve "high accuracy" just by always guessing N2, but it will completely miss N1.
            *   **The Solution (Class Weights)**: `cwN1` acts as a "penalty multiplier" in the loss function.
                *   `cwN1-8.0`: The model is penalized **8x** more for every mistake it makes on an N1 epoch compared to a standard epoch.
                *   Basically, it tells the model: *"I don't care if you get a few N2s wrong, but you better not miss the N1s!"*
            
            **Validation Loss: Is Lower (0.5533) Always Better?**
            *   **0.5533 (Model A)**: This model has a lower error on the validation set. Mathematically, it is the "better" fit for that specific dataset. It predicts the sleep stages with high confidence and accuracy.
            *   **0.6349 (Model B)**: This model has a higher error. Why? Because it was trained with **Augmentation**.
                *   **Augmentation** adds "noise" (random shifts, scaling, interference) to the training data. This makes the exam "harder" for the model (hence the higher loss).
                *   **The Trade-off**: While Model A scores better on the test, Model B might be more **robust**. Because it learned to handle noisy, augmented data, it may perform better on real-world patients with messy signals, whereas Model A might fail if the signal isn't perfect.
            *   **Conclusion**: Model A is the safe statistical winner. Model B is the specific attempt to force better N1 detection under difficult conditions.
            
            **What is "ConvNext Base"?**
            *   It is a modern **Convolutional Neural Network** architecture (2022) designed to compete with Vision Transformers. It uses large kernel sizes (7x7) to capture long-range patterns in the spectrogram, ideal for detecting sleep spindles (N2) and slow waves (N3).
            """,
                "es": """
            ### Análisis comparativo: set de entrenamiento de 2000 registros
            
            Esta carpeta contiene modelos entrenados en un subconjunto de **2000** archivos del dataset SHHS.
            
            #### **1. Mejor modelo vs. modelo con aumento de datos**
            Comparamos los dos candidatos principales en esta carpeta:
            
            | Característica | **Modelo A (Sept 4)** | **Modelo B (Sept 9)** |
            | :--- | :--- | :--- |
            | **Nombre de archivo** | `...workers2.ckpt` | `...Augmented_cwN1-6.5.ckpt` |
            | **Pérdida (Val Loss)** | **0.5533** (Menor es "Mejor") | 0.6349 |
            | **Pesos de clase** | cwN1-8.0 | **cwN1-6.5** |
            | **Aumento de datos** | Estándar | **Aumento de datos pesado** |
            
            #### **2. Conceptos educativos**
            
            **¿Qué es cwN1? (Peso de clase para N1)**
            *   **El Problema**: En datos de etapas del sueño, la etapa N1 es muy rara (típicamente ~5% de las épocas), mientras que N2 y Vigilia son abundantes. Un modelo estándar puede lograr "alta precisión" simplemente adivinando siempre N2, pero fallará completamente en N1.
            *   **La Solución (Pesos de clase)**: `cwN1` actúa como un "multiplicador de penalización" en la función de pérdida.
                *   `cwN1-8.0`: El modelo es penalizado **8 veces** más por cada error que comete en una época N1 en comparación con una época estándar.
                *   Básicamente, le dice al modelo: *"¡No me importa si te equivocas en algunos N2, pero mejor no te pierdas los N1!"*
            
            **Val Loss: ¿Es siempre mejor un valor más bajo (0.5533)?**
            *   **0.5533 (Modelo A)**: Este modelo tiene un menor error en el set de validación. Matemáticamente, es el "mejor" ajuste para ese dataset específico. Predice las etapas del sueño con alta confianza y precisión.
            *   **0.6349 (Modelo B)**: Este modelo tiene un error más alto. ¿Por qué? Porque fue entrenado con **Aumento de datos**.
                *   **El Aumento de datos** agrega "ruido" (cambios aleatorios, escalado, interferencia) a los datos de entrenamiento. Esto hace que el examen sea "más difícil" para el modelo (de ahí la pérdida más alta).
                *   **El compromiso**: Mientras que el Modelo A tiene mejor puntaje en la prueba, el Modelo B podría ser más **robusto**. Debido a que aprendió a manejar datos ruidosos y con aumento de datos, puede funcionar mejor en pacientes del mundo real con señales sucias, mientras que el Modelo A podría fallar si la señal no es perfecta.
            *   **Conclusión**: El Modelo A es el ganador estadístico seguro. El Modelo B es el intento específico de forzar una mejor detección de N1 bajo condiciones difíciles.
            
            **¿Qué es "ConvNext Base"?**
            *   Es una arquitectura moderna de **Red Neuronal Convolucional** (2022) diseñada para competir con Vision Transformers. Utiliza tamaños de kernel grandes (7x7) para capturar patrones de largo alcance en el espectrograma, ideal para detectar husos de sueño (N2) y ondas lentas (N3).
            """
            }
        }

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
        display_cols = ['filename', 'val_loss', 'group', 'model_architecture', 'params_m', 'epoch', 'size_mb', 'date_modified']
        display_cols = [c for c in display_cols if c in df_display.columns]
        
        st_df = df_display[display_cols].sort_values(by='date_modified', ascending=False) # Sort by date modified (newest first)
        
        # Apply style
        styled_df = st_df.style.apply(highlight_good_models, axis=1).format({
            "val_loss": "{:.4f}",
            "size_mb": "{:.1f}",
            "params_m": "{:.2f}"
        })
        
        st.dataframe(styled_df, width=None, use_container_width=True) # Removed height=800

        # Render Report if available for the selected group (AFTER Table)
        if selected_group in REPORTS:
            # Prepare title with translation
            # 'files' -> 'registros' is already handled in the dropdown content, but we need to do it for the title too
            current_group_title = selected_group
            if LANG == 'es':
                current_group_title = current_group_title.replace("files", "registros")
                report_title = f"📊 Informe detallado: {current_group_title}"
            else:
                report_title = f"📊 Detailed Report: {current_group_title}"

            with st.expander(report_title, expanded=True):
                # Select language content (default to English if not found)
                report_content = REPORTS[selected_group].get(LANG, REPORTS[selected_group].get("en", ""))
                st.markdown(report_content)
    else:
        st.info(t("no_checkpoints"))

# ==============================================================================
# TAB 2: INFERENCE
# ==============================================================================
with tab2:
    # Create layout: 2 Main Columns
    # Col 1: Sidebar-like File List (Full height)
    # Col 2: Main Content (Uploader/Model on top, Results below)
    # Adjusted ratios for better spacing (Ref: User Image)
    col_left_sidebar, col_main_content = st.columns([1.0, 9.0])
    
    selected_history_files = []
    
    # --- COLUMN 1: LEFT SIDEBAR (History) ---
    with col_left_sidebar:
        # Standardized Header
        st.markdown(f"**{t('history_label')}**")
        
        if processed_files:
             # Container with fixed height for scrolling (Sidebar behavior)
             # Using st.checkbox ensures clicking the name toggles selection
             with st.container(height=700, border=True):
                 # "Select All" option could be added here if needed, but keeping it simple first
                 for f in processed_files:
                     # Display name without extension for cleaner look
                     display_name = f.replace(".parquet", "").replace("_processed", "")
                     # Unique key for each checkbox
                     if st.checkbox(display_name, key=f"hist_{f}"):
                         selected_history_files.append(f)
             
             # Show count
             if selected_history_files:
                 st.caption(f"{len(selected_history_files)} selected")
        else:
            st.info("-")

    # --- COLUMN 2: MAIN CONTENT ---
    with col_main_content:
        # Top Row: Uploader and Model Selector
        # Adjusted inner ratios: Uploader needs more horizontal space
        col_top_upload, col_top_model = st.columns([2.5, 7.5])
        
        # --- Top Left: Uploader ---
        with col_top_upload:
            upload_label_markdown = t('upload_label').replace('\n', '  \n')
            st.markdown(f"**{upload_label_markdown}**")
            uploaded_raw_files = st.file_uploader(
                label=t("upload_label"),
                type=["parquet", "edf", "bdf", "xml"], # Added BDF
                accept_multiple_files=True,
                key="uploader_main",
                label_visibility="collapsed"
            )
            
            # --- SEPARATE XML and DATA FILES ---
            uploaded_files = []
            xml_files_map = {} # filename_prefix -> xml_file_content (bytes or path)
            
            if uploaded_raw_files:
                for uf in uploaded_raw_files:
                     if uf.name.lower().endswith(".xml"):
                         # Store XML for matching. Use match key logic later.
                         xml_files_map[uf.name] = uf
                     else:
                         uploaded_files.append(uf)
            
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
                    
                    info_c1, info_c2, info_c3, info_c4 = st.columns(4)
                    
                    with info_c1:
                        st.markdown(f"**{t('date_label')}:** {selected_model_row.get('date', 'N/A')}  \n"
                                    f"**{t('arch')}:** {selected_model_row.get('model_architecture', 'Unknown')}")
                    
                    with info_c2:
                        v_loss = selected_model_row.get('val_loss', 0)
                        v_loss_str = f"{v_loss:.4f}" if isinstance(v_loss, (int, float)) else f"{v_loss}"
                        st.markdown(f"**{t('time_label')}:** {selected_model_row.get('time', 'N/A')}  \n"
                                    f"**{t('val_loss')}:** {v_loss_str}")
                        
                    with info_c3:
                         st.markdown(f"**{t('files_label')}:** {n_files}  \n"
                                     f"**{t('epoch')}:** {selected_model_row.get('epoch', 'N/A')}")
                                     
                    with info_c4:
                         # Param string construction
                         fname = selected_model_row.get('filename', '')
                         import re
                         cw_match = re.search(r'(cwN1-[\d\.]+)', fname)
                         cw_str = cw_match.group(1) if cw_match else (weights if weights else "")
                         lr_val = selected_model_row.get('lr', 'N/A')
                         w_val = selected_model_row.get('workers', 0)
                         params_str = f"lr = {lr_val}, {cw_str} Workers = {w_val}"
                         
                         st.markdown(f"**{t('params_label')}:** {selected_model_row.get('params_m', 0):.2f} M  \n"
                                     f"**Hyperparams:** {params_str}")
                    
                    # User requested reduced vertical space. Using HTML hr with negative margin.
                    st.markdown("""<hr style="height:1px;border:none;color:#333;background-color:#ccc; margin-top: -5px; margin-bottom: 10px;" /> """, unsafe_allow_html=True)
                 else:
                    selected_model_name = None

        # --- GAP (User Request: ~20px) ---
        st.write("") 
        st.write("") 
        
        # --- GLOBAL STATUS AREA ---
        # User wants to see progress "here" (below uploader)
        global_status_container = st.empty()

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
                 # Use GLOBAL container for high-level status as requested
                 global_status_container.info(f"Processing {i+1}/{len(files_to_process)}: {uploaded_file.name}")
                 
                 # Keep local status for card context if needed, but rely on global for "red arrow" area
                 status_text = st.empty()
                 progress_bar = st.progress(0)
                
                 # Init scope variables
                 gt_labels = None
                
                 try:
                   # 1. Save uploaded file (Skip for History MockFiles)
                    is_history_file = hasattr(uploaded_file, 'from_history') and uploaded_file.from_history
                    
                    # Determine extension
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    # FIX: Handle History files missing extension
                    if is_history_file and not file_ext:
                        # Try to resolve actual file on disk
                        # Common pattern: shhs1-200001 -> shhs1-200001_processed.parquet
                        potential_paths = [
                            f"{uploaded_file.name}_processed.parquet",
                            f"{uploaded_file.name}.parquet"
                        ]
                        for p_path in potential_paths:
                            if os.path.exists(p_path):
                                uploaded_file.name = p_path
                                file_ext = ".parquet"
                                # st.toast(f"Resolved history file to: {p_path}")
                                break

                    temp_filename = f"temp_upload_{i}{file_ext}" # Use unique name
                    
                    if not is_history_file:
                        with open(temp_filename, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    else:
                        pass
                    
                    # EDF/BDF Conversion Logic (Only for new real uploads)
                    if file_ext in [".edf", ".bdf"] and not is_history_file:
                        global_status_container.info(f"Converting {uploaded_file.name} to Parquet...")
                        
                        # Define output parquet path
                        parquet_filename = os.path.splitext(temp_filename)[0] + ".parquet"
                        
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
                        global_status_container.success(f"Conversion complete for {uploaded_file.name}. Running inference...")

                    # Check cache for the ORIGINAL filename (edf or parquet)
                    cached_preds, cached_gts = retrieve_from_sql(uploaded_file.name)
                    
                    # Load Data immediately (only if parquet AND not history mock)
                    if file_ext == ".parquet" and not is_history_file:
                        input_df = pd.read_parquet(temp_filename)
                    elif is_history_file and file_ext == ".parquet":
                        # Try to load if persisted on disk (User Request: "Actual parquet file")
                        if os.path.exists(uploaded_file.name):
                             try:
                                 input_df = pd.read_parquet(uploaded_file.name)
                                 st.toast(f"Loaded content for {uploaded_file.name} from disk")
                             except:
                                 input_df = pd.DataFrame()
                        else:
                             input_df = pd.DataFrame()
                    else: 
                         # Fallback
                         input_df = pd.DataFrame() 

                    # --- XML / GROUND TRUTH LOOKUP ---
                    # Logic: 
                    # 1. Check if user explicitly uploaded an XML with matching name
                    # 2. Check SHHS_XML_DIRS for matching file (e.g. shhs1-200001.nsrr.xml for shhs1-200001.parquet)
                    # 3. Check if cached_gts exists from SQL (Standalone mode)
                    
                    base_name_clean = os.path.splitext(uploaded_file.name)[0].split('.')[0] # e.g. "shhs1-200001"
                    # Handle _processed suffix
                    if base_name_clean.endswith("_processed"):
                        base_name_clean = base_name_clean.replace("_processed", "")
                        
                    found_xml_path = None
                    gt_labels = None
                    
                    # A. Check Uploads
                    for xml_name, xml_file_obj in xml_files_map.items():
                        if base_name_clean in xml_name:
                             t_xml = f"temp_{xml_name}"
                             with open(t_xml, "wb") as f:
                                 f.write(xml_file_obj.getbuffer())
                             found_xml_path = t_xml
                             break
                    
                    # B. Check Local Directories
                    # Logic Update: Only rely on cached_gts if they contain at least ONE valid label (>=0). 
                    # If cached_gts is all -1, it means we processed it before without finding the XML. 
                    # In that case, we should TRY to find the XML again.
                    
                    cached_has_valid_gt = False
                    if cached_gts:
                        cached_has_valid_gt = any(g >= 0 for g in cached_gts)
                    
                    if not found_xml_path and not cached_has_valid_gt: # Look if no cache OR cache is bad
                        for search_dir in SHHS_XML_DIRS:
                            if not os.path.exists(search_dir): continue
                            candidates = glob.glob(os.path.join(search_dir, f"{base_name_clean}*.xml"))
                            if candidates:
                                found_xml_path = candidates[0]
                                break
                    
                    if found_xml_path:
                        st.caption(f"Found Ground Truth: `{os.path.basename(found_xml_path)}`")
                        gt_labels = extract_gt_from_xml(found_xml_path)
                    elif cached_gts:
                         st.caption("Found Ground Truth (SQL)")
                         gt_labels = cached_gts
                    else:
                        st.caption("No Ground Truth XML found.") 
                    
                    # --- INITIALIZE DF IF EMPTY (Standalone Mode) ---
                    # Logic: If we have GT or Preds, we need a dataframe to hold them.
                    
                    if input_df.empty:
                        if gt_labels:
                            input_df = pd.DataFrame(index=range(len(gt_labels)))
                        elif cached_preds:
                             input_df = pd.DataFrame(index=range(len(cached_preds)))

                    # --- MERGE GROUND TRUTH IF FOUND ---
                    if gt_labels and not input_df.empty:
                        # Check: Do we already have valid data?
                        has_valid_df_gt = False
                        df_gt_col = next((c for c in ['true_label', 'label', 'stage', 'sleep_stage'] if c in input_df.columns), None)
                        if df_gt_col and pd.api.types.is_numeric_dtype(input_df[df_gt_col]):
                             if (input_df[df_gt_col] >= 0).any():
                                 has_valid_df_gt = True
                        
                        # Check: Is the new GT useful?
                        new_gt_is_valid = any(x >= 0 for x in gt_labels)
                        
                        # Logic: 
                        # 1. If XML found (implies specific user intent), we usually overwrite.
                        # 2. If SQL found (cached), we only overwrite if it's better.
                        # But simplest robust logic: If new GT is invalid (all -1), DON'T overwrite valid data.
                        
                        should_merge = True
                        if has_valid_df_gt and not new_gt_is_valid:
                             should_merge = False
                             # st.toast("Using labels from file (SQL was empty)")
                        
                        if should_merge:
                            # Truncate to min length
                            min_len = min(len(input_df), len(gt_labels))
                            # Standardize on 'true_label' to avoid ambiguity downstream
                            # Map to strings so Viz works (expects "Vigilia", "N1" etc)
                            # REVERT: Mapping here breaks validation (>=0 check). Keep ints.
                            input_df['true_label'] = pd.Series(gt_labels[:min_len])
                            # 'stage' kept for backwards compatibility
                            input_df['stage'] = input_df['true_label']
                            
                        elif has_valid_df_gt and df_gt_col != 'true_label':
                             pass 
                             # Keep existing values (ints)
                             input_df['true_label'] = input_df[df_gt_col]
                    
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
                            
                        global_status_container.success(t("msg_loaded_cache").format(filename=uploaded_file.name))
                        progress_bar.empty()
                    else:
                        # ------------------------------------------------------------------
                        # NORMAL PATH: Run Inference (Real Uploads Only)
                        # ------------------------------------------------------------------
                        if file_ext == ".edf":
                            pass
                        
                        status_text = st.empty() # Reset local text
                        progress_bar = st.progress(0)
                        
                        # 2. Get Model Path (Use cached filepath to handle subdirs)
                        # model_path = os.path.join(BASE_DIR, selected_model_name) <-- BUG
                        try:
                            row = df[df['filename'] == selected_model_name].iloc[0]
                            model_path = row['filepath']
                        except:
                            # Fallback
                            model_path = os.path.join(BASE_DIR, selected_model_name)
                        
                        global_status_container.info(t("loading_model"))
                        progress_bar.progress(10)
                        
                        # Detection
                        arch = detect_architecture(model_path)
                        st.caption(f"{t('detected_arch')}: {arch}")
                        
                        model = get_model(model_name=arch, num_classes=5, pretrained=False)
                        model, _ = load_checkpoint_weights(model, model_path)
                        model.eval()
                        
                        global_status_container.info(f"{t('loading_data')} ({uploaded_file.name})")
                        progress_bar.progress(30)
                        
                        global_status_container.info(t("running_inf").format(len(input_df)))
                        
                        # Inference Loop (Batched & Vectorized for Speed)
                        with torch.no_grad():
                            # Prepare Data (Vectorized)
                            cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage', 'true_label'] if c in input_df.columns]
                            if cols_to_drop:
                                data_values = input_df.drop(columns=cols_to_drop).values
                            else:
                                data_values = input_df.values
                            
                            total = len(data_values)
                            batch_size = 64
                            predictions = []
                            
                            # Log-progress only every few batches
                            log_interval = max(1, total // (batch_size * 5))
                            
                            for batch_idx, start_idx in enumerate(range(0, total, batch_size)):
                                end_idx = min(start_idx + batch_size, total)
                                
                                # 1. Get Batch (N, 4560)
                                batch_flat = data_values[start_idx:end_idx].astype(np.float32)
                                
                                # 2. Preprocess Vectorized (Match preprocess_spectrogram logic)
                                # Mean/Std per sample (axis 1)
                                mean = batch_flat.mean(axis=1, keepdims=True)
                                std = batch_flat.std(axis=1, keepdims=True)
                                batch_norm = (batch_flat - mean) / (std + 1e-6)
                                batch_reshaped = batch_norm.reshape(-1, 1, 76, 60)
                                
                                # 3. To Tensor
                                input_tensor = torch.from_numpy(batch_reshaped)
                                
                                # 4. Forward Pass
                                logits = model(input_tensor)
                                pred_batch = torch.argmax(logits, dim=1).tolist()
                                predictions.extend(pred_batch)
                                
                                # Progress
                                if batch_idx % log_interval == 0:
                                    prog = 30 + int(60 * (end_idx / total))
                                    progress_bar.progress(prog)
                        
                        # SAVE TO SQL
                        dummy_confs = [1.0] * len(predictions)
                        # Append suffix for display in history: SC4001E.parquet -> SC4001E_processed.parquet
                        base, ext = os.path.splitext(uploaded_file.name)
                        processed_filename = f"{base}_processed{ext}"
                        
                        # PERSIST FILE TO DISK (User Request: "Actual parquet file")
                        # Copy the temp_filename (which holds the converted/uploaded data) to the processed name
                        try:
                            if os.path.exists(temp_filename):
                                shutil.copy(temp_filename, processed_filename)
                        except Exception as e:
                            print(f"Error persisting file: {e}")
                        
                        save_success = save_results_to_sql(processed_filename, predictions, dummy_confs, selected_model_name)
                        if save_success:
                            st.toast(f"Saved {processed_filename} to history!")
                            # Force sidebar update so suffix appears
                            time.sleep(1) # Small delay for toast
                            st.rerun()

                        progress_bar.progress(100)
                        global_status_container.success(f"{t('analysis_complete')}: {uploaded_file.name}")
                        progress_bar.empty()
                    
                    # Process Results
                    stage_map = {0: t("stage_wake"), 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
                    input_df['predicted_mid'] = predictions
                    input_df['predicted_label'] = input_df['predicted_mid'].map(stage_map)
                    
                    # --- SAFETY MERGE: Ensure Ground Truth is present before Viz ---
                    # If upstream logic reset input_df (e.g. History file reload), we restore GT here.
                    if 'true_label' not in input_df.columns:
                        # Try to find GT source again
                        # Keep ints for validation
                        if gt_labels:
                             min_len = min(len(input_df), len(gt_labels))
                             input_df['true_label'] = pd.Series(gt_labels[:min_len])
                             input_df['stage'] = input_df['true_label']
                        elif cached_gts:
                             min_len = min(len(input_df), len(cached_gts))
                             input_df['true_label'] = pd.Series(cached_gts[:min_len])
                             input_df['stage'] = input_df['true_label']

                    # Visualization
                    st.subheader(t("results_analysis"))
                    
                    counts = input_df['predicted_label'].value_counts().reset_index()
                    counts.columns = [t("col_stage"), t("col_count")]
                    
                    # Layout
                    # Layout: Charts Side-by-Side
                    gt_col = next((c for c in ['true_label', 'label', 'stage', 'sleep_stage'] if c in input_df.columns), None)
                    has_gt = gt_col and input_df[gt_col].notna().any()
                    
                    if has_gt:
                        # Ensure mapping exists for visualization
                        if 'true_label' not in input_df.columns:
                             input_df['true_label'] = input_df[gt_col].map(stage_map)
                    
                    if has_gt:
                        col_viz_pred, col_viz_gt = st.columns(2)
                    else:
                        col_viz_pred = st.container()
                        col_viz_gt = None
                    
                    with col_viz_pred:
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
                        ).properties(height=300)
                        st.altair_chart(chart, width="stretch")
                        
                        # Prediction Counts Row (Metrics)
                        st.markdown(f"##### {t('pred_counts')}")
                        p_counts = input_df['predicted_label'].value_counts()
                        
                        # Create 5 columns for W, N1, N2, N3, REM
                        cnt_cols = st.columns(5)
                        stages_ordered = [t("stage_wake"), "N1", "N2", "N3", "REM"]
                        
                        for idx, stage_name in enumerate(stages_ordered):
                            count_val = p_counts.get(stage_name, 0)
                            with cnt_cols[idx]:
                                st.metric(label=stage_name, value=count_val)

                    # --- Ground Truth Chart (Moved here) ---
                    if has_gt and col_viz_gt:
                         with col_viz_gt:
                             st.markdown(f"#### {t('comp_title')}") # Reusing "Ground Truth Comparison" title or similar? 
                             # User asked for "Visual Comparison" next to it.
                             
                             # Robust Valid Mask Creation
                             if pd.api.types.is_numeric_dtype(input_df[gt_col]):
                                 valid_mask = input_df[gt_col] >= 0
                             else:
                                 # Assume it's strings. Filter out potential 'invalid' strings if any, or assume all are valid?
                                 # Common invalid markers: "-1", "Unknown", "NULL", "nan"
                                 # Robust way: Check if value is in our known stages (English or Spanish)
                                 # But we might have different languages. 
                                 # Simplest: Check if not null/nan and not "-1"
                                 valid_mask = input_df[gt_col].notna() & (input_df[gt_col] != -1) & (input_df[gt_col] != "-1")

                             if not valid_mask.any():
                                 # Case: GT column exists but all values are -1 (Unknown)
                                 st.warning("Ground Truth column found but contains no valid labels (-1).")
                             else:
                                 # MAP HERE FOR VIZ (Fix for shhs1-200010 thin bars)
                                 # Create copy to avoid SettingWithCopy warning on slice
                                 viz_df = input_df.loc[valid_mask].copy()
                                 
                                 # Smart Mapping: Only map if it LOOKS like integers (0, 1, 2...)
                                 # If it's already "Wake", "N1", mapping with {0: "Wake"} will result in NaNs!
                                 if pd.api.types.is_numeric_dtype(viz_df['true_label']):
                                      viz_df['true_label'] = viz_df['true_label'].map(stage_map)
                                 
                                 gt_counts_viz = viz_df['true_label'].value_counts().reset_index()
                                 gt_counts_viz.columns = [t("col_stage"), t("col_count")]
                                 
                                 # Using Green color as seen in user screenshot for GT
                                 chart_gt = alt.Chart(gt_counts_viz).mark_bar().encode(

                                    x=alt.X(t("col_stage"), sort=sort_order, axis=alt.Axis(labelAngle=0)),
                                    y=t("col_count"),
                                    color=alt.value("#4caf50"), # Green
                                    tooltip=[t("col_stage"), t("col_count")]
                                 ).properties(height=300) # Title removed to avoid warping height
                                 
                                 st.altair_chart(chart_gt, width="stretch")
    
                                 # Ground Truth Counts Row (Metrics) - Placed under graph for comparison
                                 st.markdown(f"##### {t('comp_title')} - {t('col_count')}") 
                                 # (Previously 'Conteo de predicciones' was confusing here)
                                 # Actually user said "Add the 'Conteo de predicciones' underneath..."
                                 # It's ambiguous if they want the literal string 'Conteo de predicciones' or 'Ground Truth Counts'. 
                                 # Given the context "so they become easy to compare", it implies they want the counts of the GT.
                                 # I will use the generic 'Conteo de predicciones' label if that's what t('pred_counts') returns (it likely does), 
                                 # OR I should check strictly. Let's stick to the visual symmetry. 
                                 # However, labeling GT data as "Prediction Counts" is confusing. 
                                 # I will try to use a more accurate label if available, or just reuse the style. 
                                 # But the user asked: "Add the 'Conteo de predicciones' underneath..."
                                 # Using mapped viz_df for counts too so keys match 'stage_name' (strings)
                                 gt_val_counts = viz_df['true_label'].value_counts()
                                 
                                 gt_cnt_cols = st.columns(5)
                                 for idx, stage_name in enumerate(stages_ordered):
                                     count_val = gt_val_counts.get(stage_name, 0)
                                     with gt_cnt_cols[idx]:
                                         st.metric(label=stage_name, value=count_val)


                    
                    # --- Detailed Results Tables (Side-by-Side) ---
                    st.divider()
                    st.markdown(f"### {t('results_title')}")
                    
                    # Create 3 Columns for Tables (Preds, Comparison, Confusion Matrix)
                    tbl_col1, tbl_col2, tbl_col3 = st.columns([1, 1, 1])
                    
                    # --- LEFT TABLE: Model Predictions ---
                    with tbl_col1:
                        st.markdown(f"#### {t('pred_table_title')}")
                        
                        # Add Epoch column if not present
                        if 'Epoch' not in input_df.columns:
                            input_df['Epoch'] = range(len(input_df))
                        
                        # Display Columns
                        pred_display_cols = ['Epoch', 'predicted_label', 'predicted_mid']
                        
                        # Prepare DF for Left Alignment (Cast to string) and Select Columns
                        left_align_df = input_df[pred_display_cols].astype(str)
                        
                        st.dataframe(
                            left_align_df, 
                            use_container_width=True,  # Fit within the 3-column layout
                            height=400,
                            column_config={
                                "Epoch": st.column_config.TextColumn("Epoch"), 
                                "predicted_label": st.column_config.TextColumn(t("pred_stage")),
                                "predicted_mid": st.column_config.TextColumn(t("class_index"))
                            }
                        )
                        
                        # Download Button
                        csv = input_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=t("download_results"),
                            data=csv,
                            file_name=f"predictions_{uploaded_file.name}.csv",
                            mime="text/csv",
                            key=f"download_{i}"
                        )

                    # --- RIGHT TABLE: Ground Truth Comparison ---
                    with tbl_col2:
                         gt_col = next((c for c in ['true_label', 'label', 'stage', 'sleep_stage'] if c in input_df.columns), None)
                         
                         if gt_col:
                            st.markdown(f"#### {t('comp_title')}")
                            
                            # Map numeric GT
                            if 'true_label' not in input_df.columns:
                                 input_df['true_label'] = input_df[gt_col].map(stage_map)
                            
                            # Check explicitly for valid labels
                            valid_mask = input_df[gt_col] >= 0
                            
                            if not valid_mask.any():
                                st.warning("Ground Truth found but contains no valid labels (-1).")
                            else:
                                if valid_mask.any():
                                    correct = (input_df.loc[valid_mask, 'predicted_mid'] == input_df.loc[valid_mask, gt_col]).sum()
                                    accuracy = correct / valid_mask.sum()
                                    st.metric(t("overall_acc"), f"{accuracy:.2%}")
                                
                                # Comparison DataFrame
                                comp_df = pd.DataFrame({
                                    "Epoch": input_df['Epoch'],
                                    "Ground Truth": input_df['true_label'],
                                    "Predicted": input_df['predicted_label'],
                                    "Match": input_df[gt_col] == input_df['predicted_mid']
                                })
                                
                                st.dataframe(
                                    comp_df,
                                    use_container_width=True,
                                    height=300,
                                    column_config={
                                        "Epoch": st.column_config.NumberColumn("Epoch", format="%d"),
                                        "Match": st.column_config.CheckboxColumn("Match"),
                                        "Ground Truth": st.column_config.TextColumn("Ground Truth"),
                                        "Predicted": st.column_config.TextColumn("Predicted")
                                    }
                                )
                            
                            # Confusion Matrix moved to col 3 (below)
 
                         else:
                             st.markdown("#### Comparison")
                             # st.info("No Ground Truth labels found in this file.")
                             st.info(f"No Ground Truth labels found. Debug: Cols={input_df.columns.tolist()}, GT_Len={len(gt_labels) if gt_labels else 'None'}, Cached_SQL={bool(cached_gts)}")
                    
                    # --- COLUMN 3: CONFUSION MATRIX ---
                    with tbl_col3:
                        if gt_col and input_df[gt_col].notna().any():
                            valid_mask = input_df[gt_col] >= 0
                            if valid_mask.any():
                                st.markdown(f"#### {t('conf_matrix_title') if 'conf_matrix_title' in TRANSLATIONS else 'Confusion Matrix'}")
            
                                cm_df = pd.crosstab(
                                    input_df.loc[valid_mask, 'true_label'], 
                                    input_df.loc[valid_mask, 'predicted_label'], 
                                    rownames=['Actual'], 
                                    colnames=['Predicted']
                                )
                                # Center the matrix
                                st.dataframe(cm_df.style.background_gradient(cmap='Blues'), use_container_width=True)
                            else:
                                st.caption("No valid data for CM")
                        else:
                             st.caption("No Ground Truth")


                             
                    # --- CHARTS (Below tables or above? User asked for DETAILED RESULTS in tables) ---
                    # Ensuring charts are still present (they were earlier in code, lines ~920 in previous version)
                    # We need to make sure we didn't accidentally delete the charts by overwriting too much.
                    # Based on my previous logical block, charts were *inside* the GT block. 
                    # Providing Charts *below* the tables if GT exists to ensure completeness.
                    
                    st.divider()

                    
                    # Charts Container (Hypnogram only now)
                    c_charts = st.container()
                    
                    # Bar charts moved up.
                    
                    # Hypnogram
                    st.markdown(f"#### {t('hypno_comp') if 'hypno_comp' in TRANSLATIONS else 'Hypnogram Comparison (First 300 Epochs)'}")
                    subset = input_df.head(300).reset_index(drop=True).reset_index()
                    base = alt.Chart(subset).encode(x=alt.X('index', title="Epoch"))
                    
                    line_pred = base.mark_line(interpolate='step', color='#2196f3', size=1.5).encode(
                       y=alt.Y('predicted_mid', title="Stage", scale=alt.Scale(domain=[0, 4])),
                    )
                    
                    chart = line_pred
                    if gt_col:
                        # Prepare GT for Line Chart (Need Integers 0-4)
                        # Define robust map including Spanish
                        viz_map = {
                            'Wake': 0, 'Vigilia': 0, '0': 0,
                            'N1': 1, '1': 1,
                            'N2': 2, '2': 2,
                            'N3': 3, '3': 3,
                            'REM': 4, '4': 4
                        }
                        
                        # Apply mapping to create a temp numeric column in subset
                        # Use apply to handle mixed types safely
                        def robust_map(val):
                            if isinstance(val, (int, float)): return val
                            return viz_map.get(str(val), None)

                        subset['gt_numeric'] = subset[gt_col].apply(robust_map)
                        
                        line_gt = base.mark_line(interpolate='step', strokeDash=[5, 5], color='#4caf50', size=3).encode(
                           y=alt.Y('gt_numeric', title="Stage"),
                        )
                        chart = chart + line_gt
                        st.caption("Green (Dashed): Ground Truth | Blue (Solid): Predicted Model Output")
                    else:
                        st.caption("Blue (Solid): Predicted Model Output (No Ground Truth Available)")
                        
                    st.altair_chart(chart.properties(height=300), width="stretch")
                 except Exception as e:
                     st.error(f"Error: {e}")

# ==============================================================================
# TAB 3: BATCH PROCESSING (New Feature)
# ==============================================================================
with tab3:
    st.header(t("tab_batch"))
    
    # 1. Inputs
    # Default to a likely path or empty
    default_dir = "/Users/dhinostroza/.gemini/antigravity/scratch/tesis-app/03_nssr_shhs/parquet_files"
    source_dir = st.text_input(t("batch_source_dir"), value=default_dir)
    
    # Session State for Batch
    if 'batch_files' not in st.session_state:
        st.session_state.batch_files = []
    
    c_batch_1, c_batch_2 = st.columns([1, 4])
    
    with c_batch_1:
        if st.button(t("batch_scan_btn"), type="primary"):
            if os.path.isdir(source_dir):
                # Scan
                # Support EDF and Parquet
                edfs = glob.glob(os.path.join(source_dir, "*.edf"))
                parquets = glob.glob(os.path.join(source_dir, "*.parquet"))
                
                # Filter out those already processed
                processed_list = get_processed_files_list()
                # Normalize names (just basename)
                processed_basenames = {os.path.basename(f) for f in processed_list}
                
                pending = []
                already_processed_count = 0
                
                for fpath in edfs + parquets:
                    fname = os.path.basename(fpath)
                    # Check if processed (check both exact name and name turned into parquet)
                    name_clean = fname.replace(".edf", ".parquet")
                    
                    if fname in processed_basenames or name_clean in processed_basenames:
                        already_processed_count += 1
                    else:
                        pending.append(fpath)
                
                st.session_state.batch_files = sorted(list(set(pending)))
                st.success(f"Scan Complete.")
                
                # --- REDUNDANCY CHECK REPORT ---
                total_found = len(edfs) + len(parquets)
                processed_count = already_processed_count
                pending_count = len(st.session_state.batch_files)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Files Found", total_found)
                c2.metric("Already Processed (SQL)", processed_count)
                c3.metric("Pending Processing", pending_count)
                
                if pending_count == 0 and total_found > 0:
                    st.success("✅ INTEGRITY CHECK PASSED: All files have been processed and saved to SQL.")
                elif total_found == 0:
                    st.warning("No files found in directory.")
                else:
                    st.warning(f"⚠️ INTEGRITY CHECK AWAITING: {pending_count} files need processing.")
            else:
                st.error(t("batch_param_warning"))
                
    st.divider()
    
    # 2. Display List
    if st.session_state.batch_files:
        st.subheader(f"{t('batch_pending')} ({len(st.session_state.batch_files)})")
        df_pending = pd.DataFrame(st.session_state.batch_files, columns=["Filepath"])
        st.dataframe(df_pending, height=200, use_container_width=True)
        
        # 3. Process Button
        if st.button(t("batch_start_btn"), type="primary"):
             # Initialization
             batch_progress = st.progress(0)
             batch_status = st.empty()
             stop_btn = st.empty() # Placeholder for stop? Streamlit doesn't support interactive stop easily in loop.
             
             total_files = len(st.session_state.batch_files)
             
             # PROCESSING LOOP
             for idx, file_path in enumerate(st.session_state.batch_files):
                 fname = os.path.basename(file_path)
                 batch_status.markdown(f"**Processing {idx+1}/{total_files}:** `{fname}`")
                 
                 try:
                     # 1. Load Data
                     # Reuse logic from Inference Tab (Refactor ideally, but Copy-Paste for safety now)
                     # Determine input_df
                     current_df = pd.DataFrame()
                     
                     file_ext = os.path.splitext(fname)[1].lower()
                     
                     if file_ext in [".edf", ".bdf"]:
                        # Convert on fly
                        # batch_status.text(f"Converting {fname}...")
                        pq_path = os.path.splitext(file_path)[0] + ".parquet"
                        # Check if parquet exists?
                        if os.path.exists(pq_path):
                            current_df = pd.read_parquet(pq_path)
                        else:
                            # Run conversion
                             if 'convert_edf_to_parquet' in globals():
                                 convert_edf_to_parquet(file_path, pq_path)
                                 current_df = pd.read_parquet(pq_path)
                             else:
                                 st.error("Conversion function not found")
                                 
                     elif file_ext == ".parquet":
                        current_df = pd.read_parquet(file_path)
                        
                     # 2. Inference & Ground Truth Prep
                     if not current_df.empty:
                        # Extract Ground Truth Labels (if available in dataframe or XML)
                        # The dataframe might already have 'stage' or 'true_label' from the parquet fix
                        batch_gt = []
                        
                        # Try to get GT from columns first (efficient)
                        potential_gt_cols = ['label', 'stage', 'sleep_stage', 'true_label']
                        gt_col_found = None
                        for c in potential_gt_cols:
                            if c in current_df.columns:
                                gt_col_found = c
                                break
                        
                        if gt_col_found:
                            batch_gt = current_df[gt_col_found].tolist()
                        else:
                            # Fallback: Try XML lookup logic reused from tab2
                            # This is redundant if parquet is fixed, but robust
                            base_name_no_ext = os.path.splitext(fname)[0].replace("_processed", "")
                            # Try to find XML
                            for xml_dir in SHHS_XML_DIRS:
                                # Prioritize Profusion
                                xml_path = os.path.join(xml_dir, f"{base_name_no_ext}-profusion.xml")
                                if os.path.exists(xml_path):
                                    batch_gt = extract_gt_from_xml(xml_path)
                                    break
                                # Fallback to NSRR
                                xml_path_nsrr = os.path.join(xml_dir, f"{base_name_no_ext}-nsrr.xml")
                                if os.path.exists(xml_path_nsrr):
                                     batch_gt = extract_gt_from_xml(xml_path_nsrr)
                                     break
                                     
                        # Preprocess for Inference
                        cols_to_drop = [c for c in ['label', 'stage', 'sleep_stage', 'true_label'] if c in current_df.columns]
                        data_values = current_df.drop(columns=cols_to_drop).values if cols_to_drop else current_df.values

                        # Normalize
                        # ... (We need to ensure we use the same robust vectorization)
                        # To keep it DRY, we should have a `run_inference(df)` function.
                        # But for now, let's reuse the cached_preds logic pattern from earlier?
                        # Or better, just call the logic.
                         
                        batch_flat = data_values.astype(np.float32)
                        mean = batch_flat.mean(axis=1, keepdims=True)
                        std = batch_flat.std(axis=1, keepdims=True)
                        spectrogram_normalized = (batch_flat - mean) / (std + 1e-6)
                        spectrogram_2d = spectrogram_normalized.reshape(-1, 1, 76, 60)
                        
                        # Load Model (Ensure `model` is available or load it once)
                        # We'll reload for safety or use `st.session_state` model if available?
                        # The main script loads `model` at the top? No, it loads on selection.
                        # We need to load the BEST model for batch
                        
                        # Load Best Model
                        # Load Best Model
                        # Resolve Model Path (using same logic as Tab 2)
                        c_model_path = os.path.join(BASE_DIR, selected_model_name) # Fallback default
                        try:
                            # Use session state df to find true path (handles subdirectories)
                            row = st.session_state.df_models[st.session_state.df_models['filename'] == selected_model_name].iloc[0]
                            c_model_path = row['filepath']
                        except Exception:
                            pass

                        model = get_model(detect_architecture(c_model_path), pretrained=False)
                        model, _ = load_checkpoint_weights(model, c_model_path)
                        # model.load_state_dict(weights, strict=False) # Handled inside load_checkpoint_weights now
                        model.eval()
                        
                        # Chunked Inference
                        batch_preds = []
                        chunk_size = 64
                        with torch.no_grad():
                            t_tensor = torch.from_numpy(spectrogram_2d)
                            total_samples = len(t_tensor)
                            
                            for i in range(0, total_samples, chunk_size):
                                batch = t_tensor[i:i+chunk_size]
                                outputs = model(batch)
                                _, preds = torch.max(outputs, 1)
                                batch_preds.extend(preds.numpy().tolist())
                        
                        # 3. Save
                        # Confidence? Use dummy 1.0 or implement softmax if needed.
                        # For speed, just saving preds.
                        # Need file_id/patient_id?
                        fn_clean = fname.replace(".parquet", "").replace(".edf", "")
                        
                        # Save to SQL/Log
                        if save_results_to_sql(fname, batch_preds, [0.9]*len(batch_preds), "ConvNeXt_Tiny_v2", true_labels=batch_gt):
                            # Update batch status
                            pass
                        else:
                            st.error(f"Failed to save {fname}")
                         
                 except Exception as e:
                     st.error(f"Error processing {fname}: {e}")
                 
                 # Update Progress
                 batch_progress.progress((idx + 1) / total_files)
             
             st.success(t("batch_complete"))
             st.balloons()
             # Clear pending
             st.session_state.batch_files = []
             st.rerun() # Refresh to show in processed list

# ==============================================================================
# TAB 4: CONVERTER SCRIPT
# ==============================================================================
with tab4:
    st.header(t("tab_script"))
    script_path = "pre_shhs_edf2parquet.py"
    if os.path.exists(script_path):
        with open(script_path, "r", encoding="utf-8") as f:
            code_content = f.read()
            st.code(code_content, language="python")
    else:
        st.error(f"Script file not found: {script_path}")

# ==============================================================================
# TAB GLOSSARY
# ==============================================================================
with tab_glossary:
    import cols_glossary
    import importlib
    importlib.reload(cols_glossary)
    from cols_glossary import GLOSSARY_TERMS
    
    st.header("Glosario de Términos" if LANG == 'es' else "Glossary of Terms")
    
    st.markdown("""
    Esta sección define los términos técnicos clave utilizados en este proyecto, 
    abarcando conceptos de Aprendizaje Profundo, Medicina del Sueño e Ingeniería de Datos.
    """ if LANG == 'es' else """
    This section defines key technical terms used in this project, 
    covering Deep Learning, Sleep Medicine, and Data Engineering concepts.
    """)
    
    # Sort alphabetically by Spanish term
    sorted_terms = sorted(GLOSSARY_TERMS, key=lambda x: x['es'])
    
    # Create DataFrame for nice display
    import pandas as pd
    df_glossary = pd.DataFrame(sorted_terms)
    
    # Rename and Reorder columns for display
    if LANG == 'es':
        display_df = df_glossary[["es", "en", "desc"]]
        display_df.columns = ["Término (Español)", "Término (Inglés)", "Descripción"]
        main_col = "Término (Español)"
    else:
        display_df = df_glossary[["es", "en", "desc"]]
        display_df.columns = ["Term (Spanish)", "Term (English)", "Description"]
        main_col = "Term (Spanish)"
        
    # Render static table to remove scrollbars and show entire content
    # We use custom CSS to ensure the index and first content column stay wide enough

    
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=700)


