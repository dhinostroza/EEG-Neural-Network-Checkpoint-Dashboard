export const translations = {
    es: {
        welcome: "Modelo de aprendizaje profundo para la detección de los patrones\nde onda cerebral N1 durante los periodos de sueño.\nUniversidad Politécnica Salesiana\nTesis de grado\nDaniel Hinostroza, MD\n2026-01-08",
        welcomeDesc: "Seleccione un proyecto de la columna izquierda para ver su información técnica.",
        sidebar: {
            title: "Modelo de aprendizaje profundo para la detección de los patrones de onda cerebral N1 durante los periodos de sueño.\nUniversidad Politécnica Salesiana\nTesis de grado\nDaniel Hinostroza, MD\n2026-01-08",
            datasets: "Datasets",
            experiments: "Experimentos",
            version: "v1.1",
            // Project Items
            "dataset8": "Dataset: Sleep-EDF-8",
            "folder78": "Dataset: Sleep-edfx-78",
            "subj20": "20 sujetos",
            "subj40": "40 sujetos",
            "subj78": "78 sujetos - Orange",
            "cnn1": "Etapa 1 CNN",
            "lstm2": "Etapa 2 LSTM"
        },
        dashboard: {
            datasetInfo: "Información del Dataset",
            subjectsProcessed: "Sujetos Procesados",
            totalEpochs: "Total Muestras (Epochs)",
            spectrogramDims: "Dimensiones Espectrogramas",
            datasetVersion: "Dataset Versión",
            subjects: "Sujetos",
            samples: "Muestras (Epochs)",
            experimentNotProcessed: "Experimento No Procesado",
            dataPending: "Los datos para 78 sujetos están pendientes de procesamiento.",
            orangeMessage: "Este experimento no se pudo completar con Matlab debido a que no reconoce la GPU de Apple Silicon. El experimento se condujo en Orange.",
            stageTitles: {
                8: "Dataset: Sleep-edf-8",
                8: "Modelo Python (EEGSNet - MPS)",
                20: "Dataset (Sleep-EDFX-20)",
                41: "Modelo CNN (40 Sujetos)",
                42: "Modelo LSTM (40 Sujetos)",
                tm: "TM: Tuned Mixup (Mixup Ajustado)",
                lr: "LR: Logic Rules (Reglas Lógicas)",
                bcw: "BCW: Boosted Class Weights (Pesos de Clase Aumentados)",
                78: "Experimento (78 Sujetos)"
            },
            tabs: {
                report: "Reporte",
                charts: "Gráficos",
                analysis: "Análisis",
                comparison: "Comparación SOTA",
                scripts: "Scripts",
                dataset: "Info Dataset",
                tm: "TM: Tuned Mixup",
                lr: "LR: Logic Rules",
                bcw: "BCW: Boosted Class Weights"
            },
            compareTitle: "Comparación con el Estado del Arte (Li et al., 2022)",
            compareDesc: "Comparativa directa de métricas por clase entre Li (2022) y el modelo EEGSNet actual con el Dataset Sleep-EDF-8.",
            refTableTitle: "Tablas de Referencia (Li et al.)",
            ourModel: "Hinostroza (2025)",
            metric: "Métrica",
            refValue: "Li (2022)",
            diff: "Diferencia",
            report: {
                analysisPrelim: "Análisis Preliminar",
                analysisExp: "Análisis Experimental",
                datasetInfo: "Información del Dataset",
                methodology40: "Se entrenó un modelo utilizando un subconjunto de **40 sujetos** del dataset Sleep-EDFX-78. Se aplicó validación cruzada (Cross-Validation) para evaluar la robustez del modelo.",
                comparisonTitle: "Comparativa con Estado del Arte",
                comparisonText: "El modelo actual muestra un rendimiento competitivo frente a la literatura establecida, superando el baseline en la detección de la etapa N1.",
                configTitle: "Configuración del Experimento",
                configList: [
                    "**Optimizador:** Adam (LR: 0.001)",
                    "**Loss:** CrossEntropy (Ponderada)",
                    "**Arquitectura:** Híbrida CNN-LSTM",
                    "**Epochs:** 50 (con Early Stopping)"
                ],
                defaultText: "Reporte generado para la etapa {stage}. El sistema ha procesado los datos de entrada y generado las visualizaciones correspondientes."
            },
            confusionMatrix: "Matriz de Confusión",
            misclassification: "Clasificación Errónea",
            distribution: "Datos de distribución",
            baseline: "Base",
            abbreviationsLegend: {
                title: "Abreviaturas",
                items: {
                    logNorm: "Log-Norm: Normalización Logarítmica (Logarithmic Normalization)",
                    ca: "CA: Atención Coordinada (Coordinate Attention)",
                    ma: "MA: Aumento Mixup (Mixup Augmentation)",
                    tm: "TM: Mixup Ajustado (Tuned Mixup)",
                    lr: "LR: Reglas Lógicas (Logic Rules)",
                    bcw: "BCW: Pesos de Clase Aumentados (Boosted Class Weights, v1.2)",
                    wrs: "WRS: Muestreo Aleatorio Ponderado (Weighted Random Sampling, v1.3)"
                }
            },
            report8: {
                title: "Análisis del dataset Sleep-EDF-8 (Matlab/Web app con GPU mps para Apple Silicon)",
                desc: "Este ejercicio valida la arquitectura **EEGSNet (CNN-LSTM)**<sup>1</sup> en el subconjunto **Sleep-EDF-8**. Utiliza scripts ejecutados en **entorno Python local**, aprovechando la aceleración **MPS (Metal Performance Shaders)** en MacOS. Se entrenó el modelo EEGSNet para la clasificación de 5 etapas de sueño, utilizando **Leave-One-Subject-Out (LOSO)** para una estimación robusta. Se optó por migrar los scripts a este entorno por cuanto Matlab no reconoce la tarjeta gráfica de Apple Silicon (mps Neural Engine GPU).<br/><br/><sup>1</sup> Li, C., Qi, Y., Ding, X., Zhao, J., Sang, T., & Lee, M. (2022). A Deep Learning Method Approach for Sleep Stage Classification with EEG Spectrogram. <i>International journal of environmental research and public health</i>, 19(10), 6322. https://doi.org/10.3390/ijerph19106322",
                configTitle: "Configuración",
                configList: [
                    "**Dataset:** Sleep-EDF-8 (8 sujetos, 15 206 espectrogramas)",
                    "**Modelo:** EEGSNet (CNN + BI-LSTM + Clasificador auxiliar)",
                    "**Secuencia:** Ventana de contexto = 5 épocas",
                    "**Pérdida:** CrossEntropy + 0.5",
                    "**AuxLoss:** Ponderada",
                    "**Ejecución:** Python (MPS/Metal) - Batch 32"
                ],
                methodologyJustification: "Para la replicación y mejora del modelo de clasificación de etapas de sueño, se decidió una divergencia metodológica respecto al artículo original de Li et al. (2022) en cuanto a las pruebas de validez en el dataset Sleep-EDF-8. Li (2022) empleó una estrategia de validación cruzada de 20 pliegues a nivel de época (epoch-wise) para este subconjunto específico, lo que significa que mezclaron todas las épocas en un gran conjunto y las dividieron en 20 fragmentos aleatorios. Esto permite que las épocas del Sujeto 1 aparezcan tanto en el conjunto de entrenamiento como en el de prueba, lo que probablemente arroje resultados de precisión más altos que LOSO, pero sobreestimando potencialmente el rendimiento en el mundo real.<br/><br/>La implementación establecida en este proyecto utiliza una estrategia de validación cruzada Leave-One-Subject-Out (LOSO). Esta decisión fue deliberada y prioriza el rigor clínico y la aplicabilidad en el mundo real sobre la maximización teórica de métricas.",
                matlabContext: "Nota: El entrenamiento se hizo localmente utlizando mps. Los archivos '.m' y '.mat' de Matlab fueron readaptados para funcionar en esta aplicación web y generar los gráficos restantes presentes en el reporte.",
                note: "Los resultados se actualizan en tiempo real leyendo los archivos JSON generados por el entrenamiento.",
                archTitle: "Arquitectura y Análisis de Señales",
                spectrogramTitle: "Análisis de espectrograma (Matlab)",
                spectrogramDesc: "Visualización Tiempo-Frecuencia de una época de sueño (30s). Se observa la potencia espectral en distintas bandas de frecuencia. N1 es inherentemente 'desordenada' (messy). Fisiológicamente, N1 es la zona de transición. Se pierden las bandas Alpha fuertes y 'homogéneas' de la vigilia, pero aún no se han establecido los Husos estables (N2) o las ondas Delta (N3). Se caracteriza por frecuencias mixtas de bajo voltaje, que parecen 'ruido disperso' o 'heterogeneidad' en un espectrograma. Preprocesamiento: se utilizó la Transformación Logarítmica, que eleva el ruido de fondo de baja potencia al rango visible (las manchas azules/verdes). Esto evita que el modelo ignore detalles sutiles, pero hace que la imagen se vea 'granulada'.",
                layerTitle: "Grafo de capas (EEGSNet - Matlab)",
                layerDesc: "La arquitectura combina convoluciones separables en profundidad (Depthwise Separable Conv) para extracción de características espaciales y capas LSTM bidireccionales para capturar dependencias temporales.",
                verificationTitle: "Verificación de Entrada (Espectrogramas)",
                verificationDesc: "Confirmado: El modelo recibe Espectrogramas Multitaper de 3 canales como entrada. Forma del Tensor: (Batch, 3, 76, 60). Los datos se cargan activamente desde la fuente '.mat', se transponen a (N, 3, 76, 60) y se normalizan (Log+ZScore) antes del entrenamiento."
            },
            advancedMetrics: "Métricas Avanzadas",
            sleepStagesTitle: "Precisión de Clasificación por Etapa",
            hardware: "Recurso de Hardware",
            metrics: {
                kappa: "Kappa",
                f1: "F1 Score",
                precision: "Precisión",
                recall: "Recall"
            },
            trainingProgress: "Progreso del Entrenamiento",
            analysisCharts: {
                heatmapTitle: "Tasa de Error % Hinostroza (Heatmap)",
                countsTitle: "Conteo de Muestras Hinostroza",
                trueLabel: "Verdadero",
                predLabel: "Pred",
                rowNorm: "* Normalización por Fila (Recall/Sensibilidad)",
                epochs: "Epochs"
            },
            charts: {
                accuracyLabel: "Precisión (%)",
                lossLabel: "Pérdida",
                trainingSmoothed: "Entrenamiento (suavizado)",
                training: "Entrenamiento"
            },
            sidebarResults: {
                title: "Resultados",
                valAcc: "Precisión de validación",
                status: "Estado",
                trainingActive: "Entrenamiento Activo",
                completed: "Completado",
                currentLoss: "Pérdida Actual",
                trainingTime: "Tiempo de Entrenamiento",
                lastUpdate: "Última Actualización",
                device: "Dispositivo",
                trainingCycle: "Ciclo de Entrenamiento",
                epoch: "Época",
                iteration: "Iteración",
                otherInfo: "Otra Información",
                learningRate: "Tasa de aprendizaje"
            }
        }
    },
    en: {
        welcome: "Welcome to N1 EEG Detection",
        welcomeDesc: "Select a project from the left column to view technical information and generate documents.",
        sidebar: {
            title: "Deep Learning Model for N1 Brain Wave Pattern Detection during Sleep Periods.\nThesis - Daniel Hinostroza, MD\n2026-01-08",
            datasets: "Datasets",
            experiments: "Experiments",
            version: "v1.0.4 Beta",
            "dataset8": "Dataset: Sleep-edf-8",
            "folder78": "Dataset: Sleep-edfx-78",
            "subj20": "20 subjects",
            "subj40": "40 subjects",
            "subj78": "78 subjects - Orange",
            "cnn1": "Stage 1 CNN",
            "lstm2": "Stage 2 LSTM"
        },
        dashboard: {
            datasetInfo: "Dataset Information",
            subjectsProcessed: "Processed Subjects",
            totalEpochs: "Total Samples (Epochs)",
            spectrogramDims: "Spectrogram Dimensions",
            datasetVersion: "Dataset Version",
            subjects: "Subjects",
            samples: "Samples (Epochs)",
            experimentNotProcessed: "Experiment Not Processed",
            dataPending: "Data for 78 subjects is pending processing.",
            orangeMessage: "This experiment could not be completed with Matlab because it does not recognize the Apple Silicon GPU. The experiment was conducted in Orange.",
            stageTitles: {
                8: "Dataset: Sleep-edf-8",
                20: "Sleep-edfx-78: 20 Subjects",
                41: "Experiment: Sleep-edfx-78 (40 Subjects) - Stage 1 CNN",
                42: "Experiment: Sleep-edfx-78 (40 Subjects) - Stage 2 LSTM",
                78: "Experiment: Sleep-edfx-78 (78 Subjects) - Orange"
            },
            analysis: "Analysis",
            comparison: "SOTA Comparison",
            scripts: "Scripts",
            dataset: "Dataset Info"
        },
        compareTitle: "State-of-the-Art Comparison (Li et al., 2022)",
        compareDesc: "Direct class-wise metric comparison between the reference paper (Sleep-EDFX-8) and our current EEGSNet model.",
        refTableTitle: "Reference Tables (Li et al.)",
        ourModel: "Hinostroza (2025)",
        metric: "Metric",
        refValue: "Li (2022)",
        diff: "Difference",
        report: {
            analysisExp: "Experimental Analysis",
            analysisPrelim: "Preliminary Analysis",
            datasetInfo: "Dataset Info",
            methodology40: "This experiment uses the **Sleep-EDFx (SC40)** subset, processing 40 subjects to validate the scalability of the hybrid CNN-LSTM model.",
            comparisonTitle: "State of the Art Comparison",
            comparisonText: "The results obtained (Acc > 90%) are competitive with similar architectures like DeepSleepNet, which report performance in the 85-88% range for this sample size.",
            configTitle: "Experimental Configuration",
            configList: [
                "**Dataset:** Sleep-EDF Expanded (v1.0.0), 40 subjects.",
                "**Validation:** Partial Leave-One-Subject-Out (LOSO).",
                "**Classes:** 5 stages (W, N1, N2, N3, REM)."
            ],
            defaultText: "Metadata visualization for the processed dataset of {stage} subjects. Select 'Charts' (if available) to view training curves."
        },
        confusionMatrix: "Confusion Matrix",
        misclassification: "Misclassification",
        distribution: "Distribution",
        baseline: "Baseline",
        abbreviationsLegend: {
            title: "Comparison Legend",
            items: {
                logNorm: "Log-Norm: Logarithmic Normalization",
                ca: "CA: Coordinate Attention",
                ma: "MA: Mixup Augmentation",
                tm: "TM: Tuned Mixup",
                lr: "LR: Logic Rules",
                bcw: "BCW: Boosted Class Weights (v1.2)",
                wrs: "WRS: Weighted Random Sampling (v1.3)"
            }
        },
        report8: {
            title: "Experimental Analysis (Sleep-EDF-8)",
            desc: "This experiment validates the **EEGSNet (CNN-LSTM)**<sup>1</sup> architecture on the **Sleep-EDF-8** subset (8 subjects). The model is trained using **Leave-One-Subject-Out (LOSO)** for robust estimation.<br/><br/><sup>1</sup> Li, C., Qi, Y., Ding, X., Zhao, J., Sang, T., & Lee, M. (2022). A Deep Learning Method Approach for Sleep Stage Classification with EEG Spectrogram. <i>International journal of environmental research and public health</i>, 19(10), 6322. https://doi.org/10.3390/ijerph19106322",
            configTitle: "Configuration",
            configList: [
                "**Dataset:** Sleep-EDF (8 subjects).",
                "**Samples (Epochs):** 15,206.",
                "**Preprocessing:** Log-Transform + Z-Score Normalization (v1.1)",
                "**Model:** EEGSNet (CNN + Bi-LSTM + Auxiliary Classifier)",
                "**Sequence Training:** Context Window = 5 Epochs",
                "**Loss Function:** CrossEntropy + 0.5 * AuxLoss (Weighted)",
                "**Execution:** Python (MPS/Metal) - Batch 32"
            ],
            methodologyJustification: "For the replication and enhancement of the sleep stage classification model, a methodological divergence was decided from the original paper by Li et al. (2022) regarding validity testing on the Sleep-EDF-8 dataset. While Li (2022) employed a 20-fold epoch-wise cross-validation strategy for this specific subset, meaning that they mixed all epochs together into one big pool and split them into 20 random chunks. This allows epochs from Subject 1 to appear in both the training set and the test set likely yielding higher accuracy results than LOSO, but potentially overestimating real-world performance.<br/><br/>The implementation set in this project utilizes a Leave-One-Subject-Out (LOSO) cross-validation strategy. This decision was deliberate and prioritizes clinical rigor and real-world applicability over theoretical metric maximization.",
            matlabContext: "Originally, the processing scripts and labels (.mat) were generated in Matlab. To ensure repeatability and extend the analysis, these files were processed and adapted for this web application, enabling the generation of new charts and metrics in real-time.",
            note: "Note: Training is performed locally on this machine using the Neural Engine/GPU.",
            archTitle: "Architecture & Signal Analysis",
            spectrogramTitle: "Real N1 Spectrogram (Channel Fpz-Cz)",
            spectrogramDesc: "Visual sample (Index 1232). The model was trained on the full set of 15,206 spectrograms. N1 is inherently \"messy\". Physiologically, N1 is the transition zone. You lose the strong, \"homogeneous\" Alpha bands of wakefulness, but you haven't yet established the stable Spindles (N2) or Delta waves (N3). It is characterized by mixed low-voltage frequencies, which look like \"scattered noise\" or \"heterogeneity\" in a spectrogram. Preprocessing: the Log-Transform was used, which pulls low-power background noise up into the visible range (the blue/green speckles). This prevents the model from ignoring subtle details, but it makes the image look \"grainy.\"",
            layerTitle: "Layer Analysis (trainNetwork)",
            layerDesc: "Detailed EEGSNet structure: CNN for spatial feature extraction followed by Bi-LSTM for temporal dependencies.",
            verificationTitle: "Input Verification (Spectrograms)",
            verificationDesc: "Confirmed: The model receives 3-channel Multitaper Spectrograms as input. Input Shape: (Batch, 3, 76, 60). Data is actively loaded from the '.mat' source, transposed to (N, 3, 76, 60), and normalized (Log+ZScore) before training."
        },
        advancedMetrics: "Advanced Metrics",
        sleepStagesTitle: "Sleep Stages Classification Accuracy",
        hardware: "Hardware Resource",
        metrics: {
            kappa: "Kappa",
            f1: "F1 Score",
            precision: "Precision",
            recall: "Recall"
        },
        trainingProgress: "Training Progress",
        analysisCharts: {
            heatmapTitle: "Hinostroza Error Rate % (Heatmap)",
            countsTitle: "Hinostroza Sample Counts",
            trueLabel: "True",
            predLabel: "Pred",
            rowNorm: "* Row Normalization (Recall/Sensitivity)",
            epochs: "Epochs"
        }
    }
};
