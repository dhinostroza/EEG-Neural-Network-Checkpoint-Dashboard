# Reporte de evolución del entorno de desarrollo y scripts (agosto 2025)

Este documento detalla la evolución de los scripts y entornos de ejecución utilizados durante el desarrollo de los modelos de clasificación de etapas de sueño. Se observa una transición clara desde scripts básicos en entornos locales/estándar hacia flujos de trabajo complejos orientados a la nube (Cloud-Native) y Colab Enterprise.

## Fase 1: Preparación de datos y entorno básico
**Objetivo:** Transformación inicial de datos y organización.
**Entorno:** Google Colab Standard + Google Drive Mount.

### 1. `2025-08-10 pre_SHHS_edf2parquet.ipynb`
*   **Descripción:** Script de preprocesamiento inicial. Convierte los archivos originales EDF del dataset SHHS a formato Parquet para una lectura más eficiente.
*   **Detalles del entorno:**
    *   Montaje directo de Google Drive (`/content/drive`).
    *   Instalación básica de librerías (`mne`, `pytorch-lightning`, `timm`) vía `pip`.
    *   Manejo de advertencias de tipos de datos mixtos (Pandas/Parquet).
*   **Hito:** Establecimiento del pipeline de datos base.

### 2. `2025-08-11 Split shhs1and2.ipynb`
*   **Descripción:** Script de utilidad para la organización de archivos. Separa y mueve los archivos procesados en directorios estructurados (`shhs1_processed`, `shhs2_processed`).
*   **Detalles del entorno:**
    *   Uso de librería estándar `shutil` y `pathlib`.
    *   Ejecución simple sobre el sistema de archivos montado de Drive.

---

## Fase 2: Experimentación y búsqueda de arquitecturas
**Objetivo:** Pruebas de concepto, sintonización de hiperparámetros y comparación de modelos.
**Entorno:** Google Colab Standard/Pro (GPU T4/V100).

### 3. Scripts de experimentación (varios)
*   **Archivos:**
    *   `2025-08-17 SHHS subset_ReduceLROnPlateau scheduler_lr_tests.ipynb`
    *   `2025-08-23 Sleep Stage Model - ConvNeXt_tiny_base_ViT_and_EfficientNet B0.ipynb`
    *   `2025-08-25 Adv_model_comparison_1000files_convNeXt_tiny_and_base.ipynb`
*   **Descripción:**
    *   Transición hacia arquitecturas más complejas (ConvNeXt, ViT, EfficientNet).
    *   Implementación de schedulers dinámicos (`ReduceLROnPlateau`).
    *   Experimentación con subconjuntos de datos (100-1000 archivos) para iteración rápida.
*   **Evolución:** Inicio de la modularización del código de entrenamiento (`SleepStageClassifierLightning`).

---

## Fase 3: Optimización de rendimiento y escalamiento
**Objetivo:** Superar cuellos de botella de I/O y entrenar con el dataset completo.
**Entorno:** Google Colab Pro+ (GPU A100/High-RAM).

### 4. `2025-08-31 Sleep Stage Classif Test environment, ConvNeXt_tiny_GDrive.ipynb`
*   **Descripción:** Entrenamiento optimizado utilizando almacenamiento local efímero.
*   **Detalles del entorno:**
    *   **Estrategia de caché local:** Copia condicional de datos desde Drive hacia el disco local de la VM (`/content/local_shhs_data`) para eliminar la latencia de red de Drive durante el entrenamiento.
    *   **Verificación de hardware:** Comprobación explícita de disponibilidad de GPU (A100) y gestión de memoria.
*   **Hito:** Solución a los problemas de velocidad de dataloader ("bottleneck") típicos de entrenamientos grandes en Drive.

---

## Fase 4: Cloud-Native y Colab Enterprise (producción)
**Objetivo:** Entrenamiento a gran escala, reproducibilidad y gestión profesional de dependencias.
**Entorno:** Google Colab Enterprise + Google Cloud Storage (GCS Buckets).

### 5. `2025-08-23 sleep-stage-training-full-parquet-dataset_GColab_Enterprise.ipynb`
*   **Descripción:** Script de entrenamiento "Estado del Arte" (SOTA) preparado para el dataset completo.
*   **Detalles del entorno:**
    *   **Google Cloud Storage (GCS):** Abandono de Google Drive en favor de Buckets (`gs://shhs-sleepedfx-data-bucket`) para almacenamiento masivo y de alta velocidad.
    *   **Optimized Dataset con Metadatos:** Implementación de `OptimizedCombinedDataset` que genera y lee un archivo de metadatos (`dataset_metadata.csv`) en el bucket, evitando re-escanear miles de archivos en cada inicio.
    *   **Autenticación:** Uso de `google.colab.auth` para acceso seguro a recursos Cloud.
    *   **Arquitectura:** Soporte multi-modelo robusto (`convnext_base`, `vit_base`).

### 6. `2025-08-24 Download dependencies and upload to bucket.ipynb`
*   **Descripción:** Gestión de infraestructura "Air-gapped" / Empresarial.
*   **Detalles del entorno:**
    *   **Descarga de Fuentes:** Descarga las distribuciones fuente (`.tar.gz`, `.whl`) de todas las dependencias críticas (`pytorch-lightning`, `timm`, etc.) localmente.
    *   **Repositorio privado:** Sube estos paquetes a un bucket dedicado (`gs://shhs-sleepedfx-colab-deps/packages/`).
    *   **Propósito:** Asegurar reproducibilidad exacta y permitir instalaciones en entornos empresariales sin acceso a internet público (PyPI), garantizando que el entorno de entrenamiento sea inmutable y seguro.

---

## Resumen de la transición
1.  **Local/Drive:** Scripts simples, acceso lento a datos, uso exclusivo de CPU por incompatibilidad entre Matlab y Apple Silicon mps GPU, gestión manual.
2.  **Optimizado:** Caché en disco VM, uso de GPUs avanzadas (A100).
3.  **Enterprise/Cloud:** Almacenamiento en Buckets, I/O asíncrono, gestión de dependencias inmutable, escalado a datasets completos (TB+).
