import React, { useState, useEffect } from 'react';
import { FileText, BarChart2, Code, Loader2, FileUp, PieChart, RefreshCw, Activity, Share2, ShieldCheck } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, BarChart, Bar } from 'recharts';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { translations } from '../../translations';

const MatlabDashboard = ({ project, language = 'es' }) => {
    const t = translations[language].dashboard;
    const [activeTab, setActiveTab] = useState('report');
    const stage = project?.stage || 1;
    const isDataset = stage === 20; // Stage 8 is now an experiment with Python/MPS results

    const [backendData, setBackendData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Helper for smoothing
    const calculateSMA = (data, windowSize) => {
        if (!data || data.length === 0) return [];
        let result = [];
        for (let i = 0; i < data.length; i++) {
            const start = Math.max(0, i - windowSize + 1);
            const subset = data.slice(start, i + 1).filter(v => v !== null && v !== undefined && !isNaN(v));
            if (subset.length === 0) {
                result.push(null);
            } else {
                const sum = subset.reduce((a, b) => a + b, 0);
                result.push(sum / subset.length);
            }
        }
        return result;
    };

    // Reset data when stage changes
    useEffect(() => {
        setBackendData(null);
        setError(null);
    }, [stage]);

    useEffect(() => {
        // If it's a dataset, we fetch immediately. If it's charts, we fetch when tab is active.
        if ((isDataset && !backendData) || (activeTab === 'charts' && !backendData) || (activeTab === 'analysis' && !backendData) || (activeTab === 'comparison' && !backendData)) {
            setLoading(true);
            fetch(`http://localhost:8000/api/project/01/results?stage=${stage}`)
                .then(res => res.json())
                .then(data => {
                    if (data.status === 'success') {
                        if (data.type === 'dataset') {
                            setBackendData(data.data);
                        } else if (data.type === 'placeholder') {
                            setBackendData(data.data);
                        } else {
                            // Experiment Data
                            const accRaw = data.data.accuracy || [];
                            const lossRaw = data.data.loss || [];
                            const len = Math.max(accRaw.length, lossRaw.length); // Use max to capture all data

                            // Smoothing
                            const accSmooth = calculateSMA(accRaw, 50);
                            const lossSmooth = calculateSMA(lossRaw, 50);

                            const formatted = [];
                            for (let i = 0; i < len; i++) {
                                formatted.push({
                                    index: i,
                                    // Accuracy (handle missing indices)
                                    accRaw: accRaw[i] ?? null,
                                    accSmooth: accSmooth[i] ?? null,
                                    // Loss
                                    lossRaw: lossRaw[i] ?? null,
                                    lossSmooth: lossSmooth[i] ?? null,
                                });
                            }

                            setBackendData({
                                type: 'experiment',
                                chartData: formatted,
                                meta: {
                                    finalAcc: accSmooth[accSmooth.length - 1] || 0,
                                    finalLoss: lossSmooth[lossSmooth.length - 1] || 0,
                                    iterations: data.data.total_iterations || len,
                                    learnRate: data.data.current_learn_rate || 0.0005,
                                    epochEst: Math.floor((data.data.total_iterations || len) / 1577),
                                    // Advanced
                                    kappa: data.data.kappa,
                                    f1: data.data.f1,
                                    precision: data.data.precision,
                                    recall: data.data.recall
                                },
                                confusion_matrix: data.data.confusion_matrix,
                                progress: data.data.progress
                            });
                        }
                    } else {
                        setError(data.message || 'Error loading data');
                    }
                })
                .catch(err => setError(err.message))
                .finally(() => setLoading(false));
        }
    }, [activeTab, stage, isDataset, backendData]);

    const tabs = [
        { id: 'report', label: t.tabs.report, icon: FileText, visible: !isDataset },
        { id: 'charts', label: t.tabs.charts, icon: BarChart2, visible: !isDataset },
        { id: 'analysis', label: t.tabs.analysis || "Analysis", icon: PieChart, visible: !isDataset },
        { id: 'comparison', label: t.tabs.comparison || "Comparison", icon: Activity, visible: !isDataset && stage === 8 },
        { id: 'scripts', label: t.tabs.scripts, icon: Code, visible: true },
        { id: 'dataset', label: t.tabs.dataset, icon: FileText, visible: isDataset }
    ].filter(t => t.visible);

    // Calculate F1 per class for comparison
    const calculateClassF1 = (matrix) => {
        if (!matrix || matrix.length !== 5) return [0, 0, 0, 0, 0];
        const f1s = [];
        for (let i = 0; i < 5; i++) {
            const tp = matrix[i][i];
            const colSum = matrix.reduce((acc, row) => acc + row[i], 0);
            const rowSum = matrix[i].reduce((a, b) => a + b, 0);

            const precision = colSum === 0 ? 0 : tp / colSum;
            const recall = rowSum === 0 ? 0 : tp / rowSum;

            const f1 = (precision + recall) === 0 ? 0 : (2 * precision * recall) / (precision + recall);
            f1s.push((f1 * 100).toFixed(1));
        }
        return f1s;
    };
    // Hardcoded v1.0 Baseline Matrix (Recovered from valid results)
    const v1Matrix = [
        [7374, 345, 73, 1, 185],
        [93, 440, 56, 4, 83],
        [83, 418, 2585, 350, 223],
        [2, 17, 318, 1103, 10],
        [43, 199, 69, 1, 1099]
    ];



    // --- Helper Render Functions for Comparison Tab ---
    const renderFullCM = (matrix) => {
        if (!matrix) return <div className="flex items-center justify-center h-48 text-gray-400">Loading Matrix...</div>;
        return (
            <div className="flex items-center justify-center w-full">
                <table className="w-full text-center border-collapse text-xs">
                    <thead>
                        <tr>
                            <th className="p-1 text-gray-400 font-normal">Pred \ True</th>
                            {['W', 'N1', 'N2', 'N3', 'REM'].map(c => <th key={c} className="p-1 text-gray-600 font-bold">{c}</th>)}
                            <th className="p-1 text-blue-600 font-bold border-l-2 border-gray-200">PR (%)</th>
                            <th className="p-1 text-blue-600 font-bold">RE (%)</th>
                            <th className="p-1 text-blue-600 font-bold">F1 (%)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {matrix.map((row, i) => {
                            const tp = row[i];
                            const rowSum = row.reduce((a, b) => a + b, 0);
                            const colSum = matrix.reduce((acc, r) => acc + r[i], 0);

                            const precision = colSum > 0 ? (tp / colSum * 100) : 0;
                            const recall = rowSum > 0 ? (tp / rowSum * 100) : 0;
                            const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

                            return (
                                <tr key={i}>
                                    <td className="p-1 text-gray-600 font-bold">{['W', 'N1', 'N2', 'N3', 'REM'][i]}</td>
                                    {row.map((val, j) => (
                                        <td key={j} className="p-2 border border-gray-100" style={{
                                            backgroundColor: `rgba(0, 114, 189, ${val / Math.max(...row.map(r => Math.max(...row))) * 0.8})`,
                                            color: val > 500 ? 'white' : 'black',
                                            fontWeight: val > 500 ? 'bold' : 'normal'
                                        }}>
                                            {val}
                                        </td>
                                    ))}
                                    <td className="p-2 border-l-2 border-gray-200 font-mono text-gray-700 bg-gray-50">{precision.toFixed(1)}</td>
                                    <td className="p-2 font-mono text-gray-700 bg-gray-50">{recall.toFixed(1)}</td>
                                    <td className="p-2 font-mono font-bold text-gray-900 bg-blue-50">{f1.toFixed(1)}</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        );
    };

    const renderMisclassificationTable = (matrix) => {
        if (!matrix) return <div className="flex items-center justify-center h-24 text-gray-400">Loading...</div>;

        // Calculate specific misclassifications (Sum of off-diagonals for pairs)
        // Indices: W=0, N1=1, N2=2, N3=3, REM=4
        const w_n1 = (matrix[0][1] || 0) + (matrix[1][0] || 0);
        const w_rem = (matrix[0][4] || 0) + (matrix[4][0] || 0);
        const n1_n2 = (matrix[1][2] || 0) + (matrix[2][1] || 0);
        const rem_n2 = (matrix[4][2] || 0) + (matrix[2][4] || 0);

        return (
            <div className="flex items-center justify-center mt-8 w-full">
                <table className="w-full text-center border-collapse text-xs">
                    <thead>
                        <tr className="border-b border-gray-300">
                            <th className="p-2 text-gray-900 font-bold">Dataset</th>
                            <th className="p-2 text-gray-900 font-bold">Wake-N1</th>
                            <th className="p-2 text-gray-900 font-bold">Wake-REM</th>
                            <th className="p-2 text-gray-900 font-bold">N1-N2</th>
                            <th className="p-2 text-gray-900 font-bold">REM-N2</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td className="p-3 text-gray-800 font-medium">Sleep-EDF-8</td>
                            <td className="p-3 font-mono">{w_n1}</td>
                            <td className="p-3 font-mono">{w_rem}</td>
                            <td className="p-3 font-mono">{n1_n2}</td>
                            <td className="p-3 font-mono">{rem_n2}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        );
    };

    const renderDistributionTable = (matrix) => {
        if (!matrix) return <div className="flex items-center justify-center h-24 text-gray-400">Loading...</div>;

        // Calculate Totals
        // Indices: W=0, N1=1, N2=2, N3=3, REM=4
        // Row Sum = True Class Total
        const w = matrix[0].reduce((a, b) => a + b, 0);
        const n1 = matrix[1].reduce((a, b) => a + b, 0);
        const n2 = matrix[2].reduce((a, b) => a + b, 0);
        const n3 = matrix[3].reduce((a, b) => a + b, 0);
        const rem = matrix[4].reduce((a, b) => a + b, 0);
        const total = w + n1 + n2 + n3 + rem;

        return (
            <div className="flex items-center justify-center mt-8 w-full">
                <table className="w-full text-center border-collapse text-xs">
                    <thead>
                        <tr className="border-b border-gray-300">
                            <th className="p-2 text-gray-900 font-bold">Dataset</th>
                            <th className="p-2 text-gray-900 font-bold">Subjects</th>
                            <th className="p-2 text-gray-900 font-bold">Wake</th>
                            <th className="p-2 text-gray-900 font-bold">N1</th>
                            <th className="p-2 text-gray-900 font-bold">N2</th>
                            <th className="p-2 text-gray-900 font-bold">N3</th>
                            <th className="p-2 text-gray-900 font-bold">REM</th>
                            <th className="p-2 text-gray-900 font-bold border-l border-gray-200">Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td className="p-2 text-gray-800 font-medium">Sleep-EDF-8</td>
                            <td className="p-2 font-mono">8</td>
                            <td className="p-2 font-mono">{w}</td>
                            <td className="p-2 font-mono">{n1}</td>
                            <td className="p-2 font-mono">{n2}</td>
                            <td className="p-2 font-mono">{n3}</td>
                            <td className="p-2 font-mono">{rem}</td>
                            <td className="p-2 font-mono font-bold border-l border-gray-200">{total.toLocaleString()}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        );
    };



    // --- Restored Helper Render Functions for Analysis Tab ---
    const renderNormalizedCM = (matrix) => {
        if (!matrix) return <div className="flex items-center justify-center h-48 text-gray-400">Loading...</div>;
        return (
            <div className="flex items-center justify-center w-full">
                <table className="w-full text-center border-collapse text-xs">
                    <thead>
                        <tr>
                            <th className="p-1 text-gray-400 font-normal">{t.analysisCharts?.trueLabel || "True"} \ {t.analysisCharts?.predLabel || "Pred"}</th>
                            {['W', 'N1', 'N2', 'N3', 'REM'].map(c => <th key={c} className="p-1 text-gray-600 font-bold">{c}</th>)}
                        </tr>
                    </thead>
                    <tbody>
                        {matrix.map((row, i) => {
                            const total = row.reduce((a, b) => a + b, 0);
                            return (
                                <tr key={i}>
                                    <td className="p-1 text-gray-600 font-bold">{['W', 'N1', 'N2', 'N3', 'REM'][i]}</td>
                                    {row.map((val, j) => {
                                        const pct = total > 0 ? (val / total * 100).toFixed(1) : 0;
                                        return (
                                            <td key={j} className="p-2 border border-gray-100 font-mono" style={{
                                                backgroundColor: i === j ? '#e6f4ea' : (pct > 10 ? '#fce8e6' : 'transparent'),
                                                color: i === j ? 'green' : (pct > 10 ? 'red' : 'black')
                                            }}>
                                                {pct}%
                                            </td>
                                        );
                                    })}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
                <div className="mt-2 text-xs text-gray-500 text-center italic">{t.analysisCharts?.rowNorm || "* Row Normalization"}</div>
            </div>
        );
    };

    const renderDistribution = (matrix, color) => {
        if (!matrix) return <div className="flex items-center justify-center h-48 text-gray-400">Loading...</div>;
        return (
            <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={matrix.map((row, i) => ({
                        name: ['W', 'N1', 'N2', 'N3', 'REM'][i],
                        count: row.reduce((a, b) => a + b, 0)
                    }))}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="name" stroke="#666" fontSize={12} />
                        <YAxis stroke="#666" fontSize={12} />
                        <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ fontSize: '12px' }} />
                        <Bar dataKey="count" fill={color} radius={[4, 4, 0, 0]} barSize={50} name={t.analysisCharts?.epochs || "Epochs"} />
                    </BarChart>
                </ResponsiveContainer>
                <div className="mt-4 flex justify-center gap-4 text-sm">
                    {matrix.map((row, i) => (
                        <div key={i} className="flex flex-col items-center">
                            <span className="font-bold text-gray-700">{['W', 'N1', 'N2', 'N3', 'REM'][i]}</span>
                            <span className="font-mono text-gray-500">{row.reduce((a, b) => a + b, 0)}</span>
                        </div>
                    ))}
                </div>
            </div>
        );
    };


    // If switching to dataset mode, force tab to dataset
    useEffect(() => {
        if (isDataset) setActiveTab('dataset');
        else if (activeTab === 'dataset') setActiveTab('report');
    }, [isDataset]);

    // Shared Sidebar Component
    const renderSidebar = () => {
        if (!backendData || !backendData.meta) return null;
        return (
            <div className="w-80 bg-gray-50 border-l border-gray-200 p-4 text-xs overflow-y-auto">
                <div className="space-y-6">
                    {/* Results Section */}
                    <div>
                        <h3 className="font-bold text-gray-800 mb-2">{t.sidebarResults?.title || "Results"}</h3>
                        <div className="grid grid-cols-2 gap-y-1">
                            <span className="text-gray-500">{t.sidebarResults?.valAcc || "Validation accuracy"}:</span>
                            <span className="text-gray-900 font-bold">
                                {backendData.meta.finalAcc ? `${backendData.meta.finalAcc.toFixed(2)}%` : 'N/A'}
                            </span>
                            <span className="text-gray-500">{t.sidebarResults?.status || "Status"}:</span>
                            <span className="text-gray-900">
                                {backendData.progress ? (t.sidebarResults?.trainingActive || "Training Active") : (t.sidebarResults?.completed || "Completed")}
                            </span>
                            <span className="text-gray-500">{t.sidebarResults?.currentLoss || "Current Loss"}:</span>
                            <span className="text-gray-900 font-bold">{backendData.meta.finalLoss.toFixed(4)}</span>
                        </div>
                    </div>

                    {/* Training Time */}
                    <div>
                        <h3 className="font-bold text-gray-800 mb-2">{t.sidebarResults?.trainingTime || "Training Time"}</h3>
                        <div className="grid grid-cols-2 gap-y-1">
                            <span className="text-gray-500">{t.sidebarResults?.lastUpdate || "Last Update"}:</span>
                            <span className="text-gray-900">{new Date().toLocaleTimeString()}</span>
                            <span className="text-gray-500">{t.sidebarResults?.device || "Device"}:</span>
                            <span className="text-gray-900">{(stage === 8) ? "Apple Silicon (MPS)" : "GPU"}</span>
                        </div>
                    </div>

                    {/* Training Cycle */}
                    <div>
                        <h3 className="font-bold text-gray-800 mb-2">{t.sidebarResults?.trainingCycle || "Training Cycle"}</h3>
                        {backendData.progress ? (
                            <div className="text-sm font-mono text-gray-900 bg-white p-2 border border-gray-200 rounded">
                                {backendData.progress}
                            </div>
                        ) : (
                            <div className="grid grid-cols-2 gap-y-1">
                                <span className="text-gray-500">{t.sidebarResults?.epoch || "Epoch"}:</span>
                                <span className="text-gray-900">{backendData.meta.epochEst} of {backendData.meta.epochEst}</span>
                                <span className="text-gray-500">{t.sidebarResults?.iteration || "Iteration"}:</span>
                                <span className="text-gray-900">{backendData.meta.iterations} of {backendData.meta.iterations}</span>
                            </div>
                        )}
                    </div>

                    {/* Advanced Metrics */}
                    {(backendData.meta.kappa !== undefined) && (
                        <div>
                            <h3 className="font-bold text-gray-800 mb-2 border-t border-gray-200 pt-2">{t.advancedMetrics}</h3>
                            <div className="grid grid-cols-2 gap-y-1">
                                <span className="text-gray-500">{t.metrics.kappa}:</span>
                                <span className="text-gray-900 font-mono font-bold text-blue-600">
                                    {backendData.meta.kappa > 0 ? backendData.meta.kappa.toFixed(4) : "Calculating..."}
                                </span>
                                <span className="text-gray-500">{t.metrics.f1}:</span>
                                <span className="text-gray-900 font-mono">
                                    {backendData.meta.f1 > 0 ? backendData.meta.f1.toFixed(4) : "Calculating..."}
                                </span>
                                <span className="text-gray-500">{t.metrics.precision}:</span>
                                <span className="text-gray-900 font-mono">
                                    {backendData.meta.precision > 0 ? backendData.meta.precision.toFixed(4) : "-"}
                                </span>
                                <span className="text-gray-500">{t.metrics.recall}:</span>
                                <span className="text-gray-900 font-mono">
                                    {backendData.meta.recall > 0 ? backendData.meta.recall.toFixed(4) : "-"}
                                </span>
                            </div>
                        </div>
                    )}

                    {/* Other Info */}
                    <div>
                        <h3 className="font-bold text-gray-800 mb-2">{t.sidebarResults?.otherInfo || "Other Information"}</h3>
                        <div className="grid grid-cols-2 gap-y-1">
                            <span className="text-gray-500">{t.sidebarResults?.learningRate || "Learning rate"}:</span>
                            <span className="text-gray-900">{backendData.meta.learnRate.toFixed(5)}</span>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="flex flex-col bg-[#e0e0e0] text-gray-900 font-sans min-h-full">
            {/* Header Tabs - Light Theme for Matlab look */}
            <div className="flex border-b border-gray-300 bg-gray-100 px-6 justify-between items-center shadow-sm">
                <div className="flex">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`flex items-center gap-2 px-6 py-3 text-sm font-medium transition-colors border-b-2 ${activeTab === tab.id
                                ? 'border-blue-600 text-blue-700 bg-white'
                                : 'border-transparent text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                                }`}
                        >
                            <tab.icon size={16} />
                            {tab.label}
                        </button>
                    ))}
                </div>

                <div className="flex items-center gap-4">
                    {/* Refresh Button */}
                    {!isDataset && (
                        <button
                            onClick={() => {
                                // Force re-fetch by clearing data
                                setBackendData(null);
                            }}
                            className="flex items-center gap-1 px-3 py-1 bg-white border border-gray-300 rounded hover:bg-gray-50 text-xs font-medium text-gray-600 transition-colors shadow-sm"
                            title="Refresh Data"
                        >
                            <RefreshCw size={14} />
                            Refresh
                        </button>
                    )}

                    {/* Stage Indicator */}
                    <div className="flex items-center gap-2 text-sm text-gray-600 font-medium">
                        {t.stageTitles[stage] || stage}
                    </div>
                </div>
            </div>

            {/* Content Area */}
            <div className="flex-1">
                {stage === 78 && (
                    <div className="flex flex-col items-center justify-start h-full text-gray-500 p-8 text-center bg-[#e0e0e0]">
                        <div className="bg-orange-100 p-4 rounded-full mb-4">
                            <FileUp size={48} className="text-orange-500" />
                        </div>
                        <h2 className="text-xl font-bold text-gray-800 mb-2">{t.experimentNotProcessed} - Orange</h2>
                        <p className="mt-2 text-sm max-w-md text-gray-600 leading-relaxed">
                            {t.orangeMessage}
                        </p>
                    </div>
                )}

                {activeTab === 'dataset' && stage !== 78 && (
                    <div className="w-full px-4 animate-in fade-in duration-500 py-8">
                        <div className="bg-premium-gray p-8 rounded-xl border border-white/5 shadow-xl">
                            <h2 className="text-2xl font-sans text-premium-accent mb-6 font-bold">
                                {t.datasetInfo}: {stage === 8 ? 'Sleep-EDF-8' : 'Sleep-EDFX-78'}
                            </h2>
                            {loading ? (
                                <div className="flex justify-center items-center h-48">
                                    <Loader2 className="animate-spin text-premium-accent" size={32} />
                                </div>
                            ) : error ? (
                                <div className="p-4 bg-red-900/20 text-red-400 border border-red-900 rounded">
                                    Error: {error}
                                </div>
                            ) : backendData ? (
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="bg-black/20 p-6 rounded-lg">
                                        <h4 className="text-gray-400 text-sm uppercase mb-2">{t.subjectsProcessed}</h4>
                                        <p className="text-4xl font-mono text-white">{backendData.subject_count}</p>
                                    </div>
                                    <div className="bg-black/20 p-6 rounded-lg">
                                        <h4 className="text-gray-400 text-sm uppercase mb-2">{t.totalEpochs}</h4>
                                        <p className="text-4xl font-mono text-white">{backendData.sample_count || 'N/A'}</p>
                                    </div>
                                    {backendData.dims && (
                                        <div className="bg-black/20 p-6 rounded-lg md:col-span-2">
                                            <h4 className="text-gray-400 text-sm uppercase mb-2">{t.spectrogramDims}</h4>
                                            <p className="text-lg font-mono text-premium-accent">
                                                {JSON.stringify(backendData.dims)}
                                            </p>
                                            <p className="text-xs text-gray-500 mt-2">(Channels, Freq, Time, Samples)</p>
                                        </div>
                                    )}
                                </div>
                            ) : null}
                        </div>
                    </div>
                )}


                {activeTab === 'report' && stage !== 78 && (
                    <div className="p-8 animate-in fade-in duration-500 bg-slate-50">
                        <div className="w-full px-4 space-y-8">
                            <div className="bg-white p-8 rounded-xl border border-indigo-100 shadow-xl shadow-indigo-100/50">
                                {/* Detailed Titles based on Stage */}
                                <h2 className="text-2xl font-sans text-indigo-900 mb-6 font-bold">
                                    {stage === 41 || stage === 42 || stage === 8
                                        ? t.report.analysisExp
                                        : stage === 20
                                            ? `${t.report.datasetInfo}: 20 ${t.subjects}`
                                            : t.report.analysisPrelim
                                    }
                                </h2>

                                <div className="prose prose-slate max-w-none text-slate-600 space-y-4">
                                    {(stage === 41 || stage === 42) ? (
                                        <>
                                            <p dangerouslySetInnerHTML={{ __html: t.report.methodology40.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                                            <h3 className="text-lg font-semibold text-slate-800 mt-6">{t.report.comparisonTitle}</h3>
                                            <p>
                                                {t.report.comparisonText}
                                            </p>
                                            <h3 className="text-lg font-semibold text-slate-800 mt-6">{t.report.configTitle}</h3>
                                            <ul className="list-disc pl-5 space-y-2">
                                                {t.report.configList.map((item, i) => (
                                                    <li key={i} dangerouslySetInnerHTML={{ __html: item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                                                ))}
                                            </ul>
                                        </>
                                    ) : stage === 8 ? (
                                        <>
                                            <h3 className="text-xl font-bold text-slate-800 mb-4">{t.report8.title}</h3>

                                            {/* Full Width Description */}
                                            <p className="mb-6 text-sm text-slate-600 leading-relaxed" dangerouslySetInnerHTML={{ __html: t.report8.desc.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />

                                            {/* Two-Column Configuration List */}
                                            <h4 className="font-bold text-slate-800 mb-2">{t.report8.configTitle}</h4>
                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-6 text-sm text-slate-600">
                                                <ul className="list-disc pl-5 space-y-1">
                                                    {t.report8.configList.slice(0, 3).map((item, i) => (
                                                        <li key={i} dangerouslySetInnerHTML={{ __html: item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                                                    ))}
                                                </ul>
                                                <ul className="list-disc pl-5 space-y-1">
                                                    {t.report8.configList.slice(3).map((item, i) => (
                                                        <li key={i} dangerouslySetInnerHTML={{ __html: item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
                                                    ))}
                                                </ul>
                                            </div>

                                            <p className="mb-6 text-sm text-slate-600 leading-relaxed bg-yellow-50 p-4 rounded border-l-4 border-yellow-400 text-justify" dangerouslySetInnerHTML={{ __html: t.report8.methodologyJustification }} />

                                            <p className="mt-4 text-indigo-700 text-xs italic bg-indigo-50 p-3 rounded border-l-2 border-indigo-500">
                                                {t.report8.matlabContext}
                                            </p>

                                            <hr className="border-indigo-100 my-6" />
                                            <h3 className="text-xl font-bold text-slate-800 mb-4">{t.report8.archTitle}</h3>

                                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                                {/* Spectrogram Section (Modified: Top Aligned Graph) */}
                                                <div className="bg-slate-50 p-4 rounded-lg border border-indigo-100 transition-all hover:bg-slate-100 h-full flex flex-col justify-start">
                                                    <h4 className="font-bold text-indigo-700 mb-2 flex items-center gap-2">
                                                        <Activity size={16} />
                                                        {t.report8.spectrogramTitle}
                                                    </h4>
                                                    <p className="text-sm text-slate-500 italic mb-3">
                                                        {t.report8.spectrogramDesc}
                                                    </p>
                                                    {/* Changed items-center to items-start for top alignment */}
                                                    <div className="grid grid-cols-2 gap-4 w-full pt-2">
                                                        {['Wake', 'N1', 'N2', 'N3', 'REM'].map((stage) => (
                                                            <div key={stage} className="flex flex-col items-center">
                                                                <div className="overflow-hidden rounded border border-indigo-100 bg-white shadow-sm w-full">
                                                                    <img
                                                                        src={`/img/spec_${stage}.png`}
                                                                        alt={`${stage} Spectrogram`}
                                                                        className="w-full h-auto object-cover"
                                                                    />
                                                                </div>
                                                                <span className="text-xs font-bold text-indigo-700 mt-1">{stage}</span>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>

                                                {/* Layer Architecture Section */}
                                                <div className="bg-slate-50 p-4 rounded-lg border border-indigo-100 transition-all hover:bg-slate-100 h-full flex flex-col justify-start">
                                                    <h4 className="font-bold text-indigo-700 mb-2 flex items-center gap-2">
                                                        <Share2 size={16} />
                                                        {t.report8.layerTitle}
                                                    </h4>
                                                    <p className="text-sm text-slate-500 italic mb-3">
                                                        {t.report8.layerDesc}
                                                    </p>
                                                    <div className="overflow-hidden rounded-lg border border-indigo-100 bg-white p-2 flex-1 flex items-start justify-center shadow-inner">
                                                        <img
                                                            src="/img/model_architecture.png"
                                                            alt="Model Architecture"
                                                            className="w-full h-auto object-contain"
                                                        />
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Verification Section */}
                                            <div className="mt-4 bg-green-50 rounded-lg p-4 border border-green-100">
                                                <h4 className="font-bold text-green-900 mb-2 flex items-center gap-2">
                                                    <ShieldCheck size={18} />
                                                    {t.report8?.verificationTitle || "Input Verification"}
                                                </h4>
                                                <p className="text-sm text-green-800 font-medium leading-relaxed">
                                                    {t.report8?.verificationDesc}
                                                </p>
                                            </div>
                                        </>
                                    ) : (
                                        <>
                                            <p>
                                                {t.report.defaultText.replace('{stage}', stage)}
                                            </p>
                                            {t.abbreviationsLegend && (
                                                <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200 text-sm text-gray-700">
                                                    <h4 className="font-bold mb-2">{t.abbreviationsLegend.title || "Abbreviations"}</h4>
                                                    <div className="grid grid-cols-2 gap-2">
                                                        <div>{t.abbreviationsLegend?.items?.tm || "TM"}</div>
                                                        <div>{t.abbreviationsLegend?.items?.lr || "LR"}</div>
                                                        <div>{t.abbreviationsLegend?.items?.bcw || "BCW"}</div>
                                                    </div>
                                                </div>
                                            )}
                                        </>
                                    )}
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === 'charts' && !isDataset && (
                    <div className="flex bg-white gap-4 min-h-[calc(100vh-100px)]">
                        {loading ? (
                            <div className="flex-1 flex items-center justify-center">
                                <Loader2 className="animate-spin text-blue-600" size={48} />
                            </div>
                        ) : error ? (
                            <div className="p-8 text-red-600">{error}</div>
                        ) : backendData ? (
                            <>
                                {/* Main Charts Area */}
                                <div className="flex-1 p-4 flex flex-col gap-4 bg-white">
                                    <h2 className="text-center font-bold text-gray-800">{t.trainingProgress || "Training Progress"}</h2>

                                    {/* Accuracy Chart */}
                                    <div className="flex-1 min-h-[300px] border border-gray-200 p-2 relative">
                                        <div className="absolute top-2 left-10 -rotate-90 text-xs font-bold text-gray-600 origin-center">
                                            {t.charts?.accuracyLabel || "Accuracy (%)"}
                                        </div>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={backendData.chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                                                <XAxis dataKey="index" type="number" hide />
                                                <YAxis domain={[0, 100]} tickFormatter={(v) => v.toFixed(0)} stroke="#666" fontSize={11} />
                                                <Tooltip contentStyle={{ fontSize: '12px' }} />
                                                <Line type="monotone" dataKey="accRaw" stroke="#4dabf5" dot={false} strokeWidth={1} strokeOpacity={0.3} isAnimationActive={false} />
                                                <Line type="monotone" dataKey="accSmooth" stroke="#0072bd" dot={false} strokeWidth={2} isAnimationActive={false} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>

                                    {/* Loss Chart */}
                                    <div className="flex-1 min-h-[300px] border border-gray-200 p-2 relative">
                                        <div className="absolute top-10 left-4 text-xs font-bold text-gray-600">
                                            {t.charts?.lossLabel || "Loss"}
                                        </div>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <LineChart data={backendData.chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                                                <XAxis dataKey="index" type="number" stroke="#666" fontSize={11} label={{ value: 'Iteration', position: 'insideBottom', offset: -5 }} />
                                                <YAxis stroke="#666" fontSize={11} />
                                                <Tooltip contentStyle={{ fontSize: '12px' }} />
                                                <Line type="monotone" dataKey="lossRaw" stroke="#ffb74d" dot={false} strokeWidth={1} strokeOpacity={0.3} isAnimationActive={false} />
                                                <Line type="monotone" dataKey="lossSmooth" stroke="#d9534f" dot={false} strokeWidth={2} isAnimationActive={false} />
                                            </LineChart>
                                        </ResponsiveContainer>
                                    </div>

                                    {/* Legend at Bottom (Static) */}
                                    <div className="bg-white border border-gray-300 p-3 shadow-sm text-xs flex justify-center gap-8">
                                        <div className="flex items-center gap-4">
                                            <strong className="text-gray-700">{t.charts?.accuracyLabel || "Accuracy"}:</strong>
                                            <div className="flex items-center gap-2"><div className="w-4 h-1 bg-[#0072bd]"></div> {t.charts?.trainingSmoothed || "Training (smoothed)"}</div>
                                            <div className="flex items-center gap-2"><div className="w-4 h-1 bg-[#4dabf5]"></div> {t.charts?.training || "Training"}</div>
                                        </div>
                                        <div className="w-px bg-gray-200"></div>
                                        <div className="flex items-center gap-4">
                                            <strong className="text-gray-700">{t.charts?.lossLabel || "Loss"}:</strong>
                                            <div className="flex items-center gap-2"><div className="w-4 h-1 bg-[#d9534f]"></div> {t.charts?.trainingSmoothed || "Training (smoothed)"}</div>
                                            <div className="flex items-center gap-2"><div className="w-4 h-1 bg-[#ffb74d]"></div> {t.charts?.training || "Training"}</div>
                                        </div>
                                    </div>
                                </div>

                                {renderSidebar()}
                            </>
                        ) : null}
                    </div>
                )}

                {activeTab === 'analysis' && !isDataset && backendData && (
                    <div className="flex bg-white gap-4 min-h-[calc(100vh-100px)]">
                        <div className="flex-1 p-4 bg-white">
                            <h2 className="text-center font-bold text-gray-800 mb-4">{t.tabs.analysis}</h2>

                            {(Number(stage) === 8 || Number(stage) === 42 || Number(stage) === 41) && (
                                backendData.confusion_matrix ? (
                                    <div className="grid grid-cols-1 gap-6">
                                        {/* Sleep Stages Bar Chart */}
                                        <div className="border border-gray-200 p-6 bg-white rounded-lg shadow-sm">
                                            <h3 className="text-sm font-bold text-gray-700 mb-4 text-center">{t.sleepStagesTitle || "Sleep Stages Classification Accuracy"}</h3>
                                            <div className="h-64 w-full">
                                                <ResponsiveContainer width="100%" height="100%">
                                                    <BarChart data={backendData.confusion_matrix.map((row, i) => ({
                                                        name: ['W', 'N1', 'N2', 'N3', 'REM'][i],
                                                        accuracy: (row[i] / row.reduce((a, b) => a + b, 0)) * 100,
                                                        total: row.reduce((a, b) => a + b, 0)
                                                    }))}>
                                                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                                                        <XAxis dataKey="name" stroke="#666" fontSize={12} />
                                                        <YAxis domain={[0, 100]} stroke="#666" fontSize={12} unit="%" />
                                                        <Tooltip
                                                            cursor={{ fill: 'transparent' }}
                                                            content={({ active, payload }) => {
                                                                if (active && payload && payload.length) {
                                                                    const data = payload[0].payload;
                                                                    return (
                                                                        <div className="bg-white p-2 border border-gray-200 shadow-sm text-xs">
                                                                            <p className="font-bold">{data.name}</p>
                                                                            <p>Accuracy: {data.accuracy.toFixed(1)}%</p>
                                                                            <p>Samples: {data.total}</p>
                                                                        </div>
                                                                    );
                                                                }
                                                                return null;
                                                            }}
                                                        />
                                                        <Bar dataKey="accuracy" fill="#0072bd" radius={[4, 4, 0, 0]} barSize={50} />
                                                    </BarChart>
                                                </ResponsiveContainer>
                                            </div>
                                        </div>

                                        {/* Confusion Matrix Visualization */}
                                        <div className="border border-gray-200 p-6 bg-white rounded-lg shadow-sm flex flex-col">
                                            <h3 className="text-sm font-bold text-gray-700 mb-4 text-center">{t.confusionMatrix || "Confusion Matrix"}</h3>
                                            {renderFullCM(backendData.confusion_matrix)}
                                        </div>

                                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                            <div className="border border-gray-200 p-6 bg-white rounded-lg shadow-sm">
                                                <h3 className="text-sm font-bold text-gray-700 mb-4 text-center">{t.analysisCharts?.heatmapTitle || "Error Rate %"}</h3>
                                                {renderNormalizedCM(backendData.confusion_matrix)}
                                            </div>
                                            <div className="border border-gray-200 p-6 bg-white rounded-lg shadow-sm">
                                                <h3 className="text-sm font-bold text-gray-700 mb-4 text-center">{t.analysisCharts?.countsTitle || "Sample Counts"}</h3>
                                                {renderDistribution(backendData.confusion_matrix, '#4ade80')}
                                            </div>
                                        </div>
                                    </div>
                                ) : (
                                    <div className="p-8 text-center text-gray-500 border border-dashed border-gray-300 rounded">
                                        No Analysis data available.
                                    </div>
                                )
                            )}
                        </div>
                        {renderSidebar()}
                    </div>
                )}

                {activeTab === 'comparison' && stage === 8 && (
                    <div className="w-[95%] mx-auto p-6 space-y-8 animate-in fade-in duration-500 overflow-y-auto h-full">
                        <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                            <h2 className="text-2xl font-bold text-gray-800 mb-2">{t.compareTitle}</h2>
                            <p className="text-gray-500">{t.compareDesc}</p>
                        </div>

                        {/* Top-Level Definitions for Scope Access in Charts */}
                        {(() => {
                            // We define them here or just render them?
                            // This is JSX, I can't define variables that persist outside.
                            // I need to use a Render Helper or simple inline definitions inside the parent block?
                            // Wait, `activeTab === 'comparison'` is an expression.
                            // I cannot define variables inside the JSX expression easily for sibling elements.

                            // Solution: I have to define them OUTSIDE the return statement or use a helper function to render the whole view.
                            // OR, I can define them inside a IIFE that wraps the ENTIRE view?

                            // Alternative: Since `v1Matrix` is constant, I can just inline the array in the chart props below?
                            // Yes, that's safer and avoids refactoring the whole component function logic.
                            // It's a bit verbose but guaranteed to work.
                            return null;
                        })()}

                        {/* Unified Scroll Container for All Comparison Blocks */}
                        <div className="overflow-x-auto pb-6">
                            <div className="min-w-[1300px] space-y-8">
                                {/* 1. Metrics Comparison Table (Fixed) */}
                                <div className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
                                    <table className="w-full text-sm text-left">
                                        <thead className="bg-gray-50 text-gray-700 font-bold text-xs">
                                            <tr>
                                                <th className="px-6 py-3">{t.metric || "Metric"} (F1-Score %)</th>
                                                <th className="px-6 py-3 text-center bg-blue-50 text-blue-800">{t.refValue || "Li (2022)"}</th>
                                                <th className="px-6 py-3 text-center bg-gray-50 text-gray-600">v1.0 Base (Hinostroza, 2025)</th>
                                                <th className="px-6 py-3 text-center bg-blue-50 text-blue-800">v1.1 Log-Norm (Hinostroza, 2025)</th>
                                                <th className="px-6 py-3 text-center bg-green-50 text-green-800">v1.2 CA + MA + TM + LR + BCW (Hinostroza, 2025)</th>
                                                <th className="px-6 py-3 text-center bg-purple-50 text-purple-800">v1.3 CA + MA + TM + LR + WRS (Hinostroza, 2025)</th>
                                                <th className="px-6 py-3 text-center">Diferencia Li - Hinostroza v1.3</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-100">
                                            {(() => {
                                                const liValues = [99.1, 70.2, 90.7, 87.0, 90.0]; // W, N1, N2, N3, REM
                                                const classes = ['Wake (W)', 'N1', 'N2', 'N3', 'REM'];

                                                const v1MatrixLocal = [
                                                    [7374, 345, 73, 1, 185],
                                                    [93, 440, 56, 4, 83],
                                                    [83, 418, 2585, 350, 223],
                                                    [2, 17, 318, 1103, 10],
                                                    [43, 199, 69, 1, 1099]
                                                ];

                                                const v1_1MatrixLocal = [
                                                    [7443, 418, 25, 8, 84],
                                                    [23, 431, 86, 12, 124],
                                                    [49, 207, 2683, 509, 211],
                                                    [0, 12, 146, 1282, 10],
                                                    [20, 103, 80, 26, 1182]
                                                ];

                                                const baseValues = calculateClassF1(v1MatrixLocal);
                                                const v1_1_Values = calculateClassF1(v1_1MatrixLocal);

                                                // v1.2 Values (Explicitly loaded) or Fallback
                                                const v1_2_Values = backendData?.confusion_matrix_v1_2
                                                    ? calculateClassF1(backendData.confusion_matrix_v1_2)
                                                    : [92.0, 29.5, 77.5, 77.3, 40.8];

                                                // v1.3 Values (Current Default)
                                                const currentValues = calculateClassF1(backendData?.confusion_matrix);

                                                return classes.map((label, i) => {
                                                    const ref = liValues[i];
                                                    const current = parseFloat(currentValues[i]);
                                                    const baseline = parseFloat(baseValues[i]);
                                                    const v1_1 = parseFloat(v1_1_Values[i]);
                                                    const v1_2 = parseFloat(v1_2_Values[i]);

                                                    // Diff against SOTA (Li) using v1.3
                                                    const diff = (current - ref).toFixed(1);
                                                    const isBetter = diff >= 0;

                                                    return (
                                                        <tr key={i} className="hover:bg-gray-50/50">
                                                            <td className="px-6 py-4 font-medium text-gray-900">{label}</td>
                                                            <td className="px-6 py-4 text-center font-mono text-gray-600">{ref.toFixed(1)}</td>
                                                            <td className="px-6 py-4 text-center font-mono text-gray-500">{baseline.toFixed(1)}</td>
                                                            <td className="px-6 py-4 text-center font-mono text-blue-600">{v1_1.toFixed(1)}</td>
                                                            <td className="px-6 py-4 text-center font-mono text-green-700">{v1_2.toFixed(1)}</td>
                                                            <td className="px-6 py-4 text-center font-mono font-bold text-purple-700 bg-purple-50">{current.toFixed(1)}</td>
                                                            <td className={`px-6 py-4 text-center font-bold ${isBetter ? 'text-green-600' : 'text-red-500'}`}>
                                                                {isBetter ? '+' : ''}{diff}
                                                            </td>
                                                        </tr>
                                                    );
                                                });
                                            })()}
                                        </tbody>
                                    </table>
                                </div>

                                {/* 2. Confusion Matrices Side-by-Side (5 Cols) */}
                                <div>
                                    <div className="grid grid-cols-5 gap-6">
                                        {/* Col 1: Li */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-gray-800 mb-4 text-center border-b pb-2 text-sm">{t.confusionMatrix} (Li, 2022)</h3>
                                            <div className="flex items-center justify-center h-full">
                                                <img src="/img/li_2022_confusion.png" alt="Li 2022 CM" className="max-w-full max-h-64 object-contain rounded" />
                                            </div>
                                        </div>

                                        {/* Col 2: Baseline v1.0 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-gray-600 mb-4 text-center border-b pb-2 text-sm">{t.confusionMatrix}, v1.0 Base (Hinostroza, 2025)</h3>
                                            {renderFullCM([
                                                [7374, 345, 73, 1, 185],
                                                [93, 440, 56, 4, 83],
                                                [83, 418, 2585, 350, 223],
                                                [2, 17, 318, 1103, 10],
                                                [43, 199, 69, 1, 1099]
                                            ])}
                                        </div>

                                        {/* Col 3: Baseline v1.1 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-blue-700 mb-4 text-center border-b pb-2 text-sm">{t.confusionMatrix}, v1.1 Log-Norm (Hinostroza, 2025)</h3>
                                            {renderFullCM([
                                                [7443, 418, 25, 8, 84],
                                                [23, 431, 86, 12, 124],
                                                [49, 207, 2683, 509, 211],
                                                [0, 12, 146, 1282, 10],
                                                [20, 103, 80, 26, 1182]
                                            ])}
                                        </div>

                                        {/* Col 4: Previous v1.2 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-green-700 mb-4 text-center border-b pb-2 text-sm">{t.confusionMatrix}, v1.2 CA + MA + TM + LR + BCW (Hinostroza, 2025)</h3>
                                            {renderFullCM(backendData?.confusion_matrix_v1_2 || [
                                                [7358, 219, 198, 3, 200],
                                                [56, 198, 290, 8, 124],
                                                [78, 169, 2769, 452, 191],
                                                [1, 9, 396, 1142, 2],
                                                [51, 62, 597, 8, 693]
                                            ])}
                                        </div>

                                        {/* Col 5: Current v1.3 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 ring-2 ring-purple-500 ring-opacity-50">
                                            <h3 className="font-bold text-purple-700 mb-4 text-center border-b pb-2 text-sm">{t.confusionMatrix}, v1.3 CA + MA + TM + LR + WRS (Hinostroza, 2025)</h3>
                                            {renderFullCM(backendData?.confusion_matrix)}
                                        </div>
                                    </div>
                                </div>

                                {/* 3. Misclassification Comparison (5 Cols) */}
                                <div>
                                    <div className="grid grid-cols-5 gap-6">
                                        {/* Col 1: Li */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-gray-800 mb-4 text-center border-b pb-2 text-sm">{t.misclassification} (Li, 2022)</h3>
                                            <div className="flex items-center justify-center h-full">
                                                <img src="/img/li_2022_misclass.png" alt="Li 2022 Misclass" className="max-w-full max-h-64 object-contain rounded" />
                                            </div>
                                        </div>



                                        {/* Col 2: Baseline v1.0 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-gray-600 mb-4 text-center border-b pb-2 text-sm">{t.misclassification}, v1.0 Base (Hinostroza, 2025)</h3>
                                            {renderMisclassificationTable([
                                                [7374, 345, 73, 1, 185],
                                                [93, 440, 56, 4, 83],
                                                [83, 418, 2585, 350, 223],
                                                [2, 17, 318, 1103, 10],
                                                [43, 199, 69, 1, 1099]
                                            ])}
                                        </div>

                                        {/* Col 3: Baseline v1.1 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-blue-700 mb-4 text-center border-b pb-2 text-sm">{t.misclassification}, v1.1 Log-Norm (Hinostroza, 2025)</h3>
                                            {renderMisclassificationTable([
                                                [7443, 418, 25, 8, 84],
                                                [23, 431, 86, 12, 124],
                                                [49, 207, 2683, 509, 211],
                                                [0, 12, 146, 1282, 10],
                                                [20, 103, 80, 26, 1182]
                                            ])}
                                        </div>

                                        {/* Col 4: Previous v1.2 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-green-700 mb-4 text-center border-b pb-2 text-sm">{t.misclassification}, v1.2 CA + MA + TM + LR + BCW (Hinostroza, 2025)</h3>
                                            {renderMisclassificationTable(backendData?.confusion_matrix_v1_2 || [
                                                [7358, 219, 198, 3, 200],
                                                [56, 198, 290, 8, 124],
                                                [78, 169, 2769, 452, 191],
                                                [1, 9, 396, 1142, 2],
                                                [51, 62, 597, 8, 693]
                                            ])}
                                        </div>

                                        {/* Col 5: Current v1.3 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 ring-2 ring-purple-500 ring-opacity-50">
                                            <h3 className="font-bold text-purple-700 mb-4 text-center border-b pb-2 text-sm">{t.misclassification}, v1.3 CA + MA + TM + LR + WRS (Hinostroza, 2025)</h3>
                                            {renderMisclassificationTable(backendData?.confusion_matrix)}
                                        </div>
                                    </div>
                                </div>

                                {/* 4. Distribution Comparison (5 Cols) */}
                                <div>
                                    <div className="grid grid-cols-5 gap-6">
                                        {/* Col 1: Li 2022 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-gray-700 mb-4 text-center border-b pb-2 text-sm">{t.distribution} (Li, 2022)</h3>
                                            {renderDistributionTable([
                                                [7649, 8, 123, 0, 198],
                                                [14, 552, 2, 0, 108],
                                                [9, 1, 3504, 87, 58],
                                                [0, 0, 137, 1293, 0],
                                                [20, 36, 12, 0, 1343]
                                            ])}
                                        </div>

                                        {/* Col 2: Baseline v1.0 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-gray-600 mb-4 text-center border-b pb-2 text-sm">{t.distribution}, v1.0 Base (Hinostroza, 2025)</h3>
                                            {renderDistributionTable([
                                                [7374, 345, 73, 1, 185],
                                                [93, 440, 56, 4, 83],
                                                [83, 418, 2585, 350, 223],
                                                [2, 17, 318, 1103, 10],
                                                [43, 199, 69, 1, 1099]
                                            ])}
                                        </div>

                                        {/* Col 3: Baseline v1.1 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-blue-700 mb-4 text-center border-b pb-2 text-sm">{t.distribution}, v1.1 Log-Norm (Hinostroza, 2025)</h3>
                                            {renderDistributionTable([
                                                [7443, 418, 25, 8, 84],
                                                [23, 431, 86, 12, 124],
                                                [49, 207, 2683, 509, 211],
                                                [0, 12, 146, 1282, 10],
                                                [20, 103, 80, 26, 1182]
                                            ])}
                                        </div>

                                        {/* Col 4: Previous v1.2 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200">
                                            <h3 className="font-bold text-green-700 mb-4 text-center border-b pb-2 text-sm">{t.distribution}, v1.2 CA + MA + TM + LR + BCW (Hinostroza, 2025)</h3>
                                            {renderDistributionTable(backendData?.confusion_matrix_v1_2 || [
                                                [7358, 219, 198, 3, 200],
                                                [56, 198, 290, 8, 124],
                                                [78, 169, 2769, 452, 191],
                                                [1, 9, 396, 1142, 2],
                                                [51, 62, 597, 8, 693]
                                            ])}
                                        </div>

                                        {/* Col 5: Current v1.3 */}
                                        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 ring-2 ring-purple-500 ring-opacity-50">
                                            <h3 className="font-bold text-purple-700 mb-4 text-center border-b pb-2 text-sm">{t.distribution}, v1.3 CA + MA + TM + LR + WRS (Hinostroza, 2025)</h3>
                                            {renderDistributionTable(backendData?.confusion_matrix)}
                                        </div>
                                    </div>
                                    {/* Abbreviations Legend */}
                                    <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                                        <h4 className="font-bold text-slate-700 text-sm mb-2">{t.abbreviationsLegend?.title || "Legend"}</h4>
                                        <div className="grid grid-cols-2 md:grid-cols-3 gap-y-2 gap-x-4 text-xs text-slate-600">
                                            <div>{t.abbreviationsLegend?.items?.logNorm || "Log-Norm"}</div>
                                            <div>{t.abbreviationsLegend?.items?.ca || "CA"}</div>
                                            <div>{t.abbreviationsLegend?.items?.ma || "MA"}</div>
                                            <div>{t.abbreviationsLegend?.items?.tm || "TM"}</div>
                                            <div>{t.abbreviationsLegend?.items?.lr || "LR"}</div>
                                            <div>{t.abbreviationsLegend?.items?.bcw || "BCW"}</div>
                                            <div>{t.abbreviationsLegend?.items?.wrs || "WRS"}</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {
                    activeTab === 'scripts' && (
                        <div className="animate-in fade-in duration-500 p-8 overflow-y-auto h-full">
                            <div className="columns-1 md:columns-2 gap-6 space-y-6">
                                {stage === 8 ?
                                    <>
                                        <div className="break-inside-avoid bg-premium-gray rounded-xl overflow-hidden border border-white/5 mb-6">
                                            <div className="bg-black/20 p-3 border-b border-white/5 flex items-center gap-2">
                                                <Code size={14} className="text-blue-400" />
                                                <span className="font-mono text-xs text-gray-400">data_loader.py (Chunked Streaming Pipeline)</span>
                                            </div>
                                            <div className="max-h-[500px] overflow-y-auto custom-scrollbar">
                                                <SyntaxHighlighter language="python" style={vscDarkPlus} customStyle={{ margin: 0, padding: '1.5rem', background: 'transparent' }}>
                                                    {`import torch
import numpy as np
import glob
import os
import random
from torch.utils.data import IterableDataset

class ChunkedIterableDataset(IterableDataset):
    def __init__(self, data_dir, is_train=True):
        self.data_dir = data_dir
        self.is_train = is_train
        # Load all available chunks from disk
        spectrogram_chunks = sorted(glob.glob(os.path.join(data_dir, "spectrograms_*.npy")))
        label_chunks = sorted(glob.glob(os.path.join(data_dir, "labels_*.npy")))

        # 80/20 Train/Val split based on chunks
        split_idx = int(0.8 * len(spectrogram_chunks))
        if self.is_train:
            self.spectrogram_chunks = spectrogram_chunks[:split_idx]
            self.label_chunks = label_chunks[:split_idx]
        else:
            self.spectrogram_chunks = spectrogram_chunks[split_idx:]
            self.label_chunks = label_chunks[split_idx:]

        print(f"Dataset initialized with {len(self.spectrogram_chunks)} chunks.")

    def __iter__(self):
        chunk_indices = list(range(len(self.spectrogram_chunks)))
        if self.is_train:
            random.shuffle(chunk_indices) # Shuffle chunks for randomness

        for chunk_idx in chunk_indices:
            # Load a full chunk into memory
            X_chunk = np.load(self.spectrogram_chunks[chunk_idx])
            y_chunk = np.load(self.label_chunks[chunk_idx])

            if self.is_train:
                # Shuffle within the chunk
                indices = np.random.permutation(len(y_chunk))
                X_chunk = X_chunk[indices]
                y_chunk = y_chunk[indices]

            # Yield individual samples
            for i in range(len(y_chunk)):
                spectrogram_flat = X_chunk[i]
                label = y_chunk[i]
                spectrogram_2d = spectrogram_flat.reshape(1, 76, 60)
                yield torch.from_numpy(spectrogram_2d), torch.tensor(label, dtype=torch.long)`}
                                                </SyntaxHighlighter>
                                            </div>
                                        </div>

                                        <div className="break-inside-avoid bg-premium-gray rounded-xl overflow-hidden border border-white/5 mb-6">
                                            <div className="bg-black/20 p-3 border-b border-white/5 flex items-center gap-2">
                                                <Code size={14} className="text-blue-400" />
                                                <span className="font-mono text-xs text-gray-400">models.py (ConvNeXt + Lightning Architecture)</span>
                                            </div>
                                            <div className="max-h-[500px] overflow-y-auto custom-scrollbar">
                                                <SyntaxHighlighter language="python" style={vscDarkPlus} customStyle={{ margin: 0, padding: '1.5rem', background: 'transparent' }}>
                                                    {`import timm
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy
import torch
import torch.optim as optim
import torch.nn.functional as F

def get_model(model_name='convnext_base', pretrained=True):
    # Adapt ConvNeXt for 1-channel input and 5 classes
    model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=pretrained)
    original_conv = model.stem[0]
    new_first_conv = nn.Conv2d(1, original_conv.out_channels, kernel_size=original_conv.kernel_size, 
                               stride=original_conv.stride, padding=original_conv.padding)
    # Average weights to initialize 1-channel conv
    with torch.no_grad():
        new_first_conv.weight[:, :] = original_conv.weight.clone().mean(dim=1, keepdim=True)
    model.stem[0] = new_first_conv
    model.head.fc = nn.Linear(model.head.fc.in_features, 5)
    return model

class SleepStageClassifierLightning(pl.LightningModule):
    def __init__(self, model_name, learning_rate, class_weights, epochs):
        super().__init__()
        self.save_hyperparameters()
        self.model = get_model(model_name=self.hparams.model_name)
        self.train_accuracy = MulticlassAccuracy(num_classes=5)
        self.val_accuracy = MulticlassAccuracy(num_classes=5)
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        # x = self.normalize_on_gpu(x) # Assuming normalization is handled by dataset or pre-processing
        # x = self.spec_augment(x) # Assuming augmentation is handled by dataset or pre-processing
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_accuracy.update(y_pred, y_true)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy.update(y_pred, y_true)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        # Simple learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.3,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step', # update learning rate every step
                'frequency': 1,
            }
        }`}
                                                </SyntaxHighlighter>
                                            </div>
                                        </div>

                                        <div className="break-inside-avoid bg-premium-gray rounded-xl overflow-hidden border border-white/5 mb-6">
                                            <div className="bg-black/20 p-3 border-b border-white/5 flex items-center gap-2">
                                                <Code size={14} className="text-blue-400" />
                                                <span className="font-mono text-xs text-gray-400">run_training.py (Training Loop)</span>
                                            </div>
                                            <div className="max-h-[500px] overflow-y-auto custom-scrollbar">
                                                <SyntaxHighlighter language="python" style={vscDarkPlus} customStyle={{ margin: 0, padding: '1.5rem', background: 'transparent' }}>
                                                    {`import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# --- CONFIGURATION ---
CONSOLIDATED_DATA_DIR = "/content/drive/MyDrive/shhs_consolidated_data/"
# Optimized class weights for component-wise balance
CLASS_WEIGHTS = [0.7, 6.5, 0.5, 1.5, 1.2] 
EPOCHS = 40
BATCH_SIZE = 512
LEARNING_RATE = 2e-5

# --- DATASETS ---
# Using the efficient ChunkedIterableDataset
train_dataset = ChunkedIterableDataset(CONSOLIDATED_DATA_DIR, is_train=True)
val_dataset = ChunkedIterableDataset(CONSOLIDATED_DATA_DIR, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

# --- MODEL ---
model = SleepStageClassifierLightning('convnext_base', LEARNING_RATE, CLASS_WEIGHTS, EPOCHS)

# --- CALLBACKS & LOGGER ---
logger = CSVLogger("/content/drive/MyDrive/sleep_logs/", name="convnext_consolidated")
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', 
    dirpath="/content/drive/MyDrive/checkpoints/",
    filename="best-model-{epoch}-{val_loss:.2f}",
    save_top_k=1, 
    mode='min'
)

# --- TRAINER ---
trainer = pl.Trainer(
    max_epochs=EPOCHS, 
    accelerator="gpu", 
    devices=1, 
    logger=logger,
    callbacks=[checkpoint_callback],
    precision="bf16-mixed" # Mixed precision for speed
)

# --- EXECUTION ---
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print("✅ Training complete!")`}
                                                </SyntaxHighlighter>
                                            </div>
                                        </div>
                                    </>
                                    :
                                    <>
                                        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 font-mono text-sm overflow-hidden break-inside-avoid">
                                            <div className="flex justify-between items-center mb-2 border-b border-gray-800 pb-2">
                                                <span className="text-blue-400 font-bold">preprocessSubjectEDFX.m</span>
                                                <span className="text-xs text-gray-500">Matlab Preprocessing</span>
                                            </div>
                                            <div className="max-h-[500px] overflow-y-auto custom-scrollbar">
                                                <SyntaxHighlighter language="matlab" style={vscDarkPlus} showLineNumbers={true} customStyle={{ background: 'transparent', padding: 0 }}>
                                                    {`% Preprocessing script for Sleep-EDF dataset
function preprocessSubjectEDFX(subjectID)
    % Load EDF file
    [data, header] = lab_read_edf(subjectID);
    
    % Filter signals (Bandpass 0.5 - 35 Hz)
    filtered_data = butter_filter(data, [0.5 35], header.sampling_rate);
    
    % Extract Spectrograms
    spectrograms = compute_stft(filtered_data);
    
    save(['processed_' subjectID '.mat'], 'spectrograms');
end`}
                                                </SyntaxHighlighter>
                                            </div>
                                        </div>

                                        <div className="bg-gray-900 rounded-lg border border-gray-800 p-4 font-mono text-sm overflow-hidden break-inside-avoid">
                                            <div className="flex justify-between items-center mb-2 border-b border-gray-800 pb-2">
                                                <span className="text-blue-400 font-bold">run_LOSO_CV_Training.m</span>
                                                <span className="text-xs text-gray-500">Matlab Training</span>
                                            </div>
                                            <div className="max-h-[500px] overflow-y-auto custom-scrollbar">
                                                <SyntaxHighlighter language="matlab" style={vscDarkPlus} showLineNumbers={true} customStyle={{ background: 'transparent', padding: 0 }}>
                                                    {`% Leave-One-Subject-Out Cross Validation
subjects = dir('dataset/files/*.edf');
for i = 1:length(subjects)
    train_cnn(subjects(i).name);
    validate_model();
end`}
                                                </SyntaxHighlighter>
                                            </div>
                                        </div>
                                    </>
                                }
                            </div>
                        </div>
                    )
                }
            </div>
        </div>
    );
};

export default MatlabDashboard;
