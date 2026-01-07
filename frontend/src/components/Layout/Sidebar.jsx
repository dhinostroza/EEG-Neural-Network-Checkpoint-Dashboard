import React, { useState } from 'react';
import { Layout, FileUp, Folder, ChevronDown, ChevronRight, Globe, Database, Beaker, FolderOpen } from 'lucide-react';
import { translations } from '../../translations';

const Sidebar = ({ onSelectProject, language = 'es', setLanguage }) => {
    const t = translations[language];

    const [projects] = useState({
        '2025-04-22': [
            {
                id: 4,
                name: '01_matlab_eeg',
                experiments: [
                    { id: 8, nameKey: 'dataset8', name: 'Dataset: Sleep-EDF-8', type: 'dataset', stage: 8 },
                    {
                        id: 'folder_78',
                        nameKey: 'folder78',
                        name: 'Dataset: Sleep-edfx-78',
                        type: 'folder',
                        children: [
                            { id: 20, nameKey: 'subj20', name: 'Dataset: Sleep-EDFX-20', type: 'dataset', stage: 20 },
                            {
                                id: 'folder_40',
                                nameKey: 'subj40',
                                name: '40 Subjects',
                                type: 'folder',
                                children: [
                                    { id: 41, nameKey: 'cnn1', name: 'Stage 1 CNN', type: 'experiment', stage: 41 },
                                    { id: 42, nameKey: 'lstm2', name: 'Stage 2 LSTM', type: 'experiment', stage: 42 }
                                ]
                            },
                            { id: 78, nameKey: 'subj78', name: '78 Subjects - Orange', type: 'placeholder', stage: 78 }
                        ]
                    }
                ]
            }
        ],
        '2025-08-12': [
            { id: 1, name: 'Analisis de Datos' },
            { id: 2, name: 'Prediccion de Ventas' }
        ],
        '2025-09-01': [
            { id: 3, name: 'Clasificacion de Imagenes' }
        ]
    });

    // State for active selection
    const [activeProject, setActiveProject] = useState(null);

    const onProjectSelect = (item) => {
        setActiveProject(item.id);
        onSelectProject(item);
    };

    const renderProjectList = (items) => (
        <div className="space-y-1">
            {items.map((item) => (
                <button
                    key={item.id}
                    onClick={() => {
                        onProjectSelect(item);
                    }}
                    className={`w-full flex items-center gap-3 px-3 py-2 text-sm transition-all rounded-md group ${activeProject === item.id
                        ? 'bg-indigo-100 text-indigo-700 shadow-sm font-medium'
                        : 'text-slate-600 hover:bg-white/60 hover:text-indigo-600'
                        }`}
                >
                    <FolderOpen size={16} className={`transition-colors ${activeProject === item.id ? 'text-indigo-600' : 'text-slate-400 group-hover:text-indigo-500'}`} />
                    <span className="truncate">{item.name}</span>
                    {activeProject === item.id && (
                        <div className="ml-auto w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
                    )}
                </button>
            ))}
        </div>
    );

    return (
        <div className="w-64 bg-gradient-to-b from-slate-50 to-indigo-50 border-r border-indigo-100 flex flex-col h-full shadow-lg z-10 transition-colors duration-300">
            {/* Header */}
            <div className="p-6 border-b border-indigo-100 flex flex-col items-center text-center gap-4 bg-white/30 backdrop-blur-sm">
                <div className="w-16 h-16 flex items-center justify-center bg-white rounded-full p-2 shadow-md ring-1 ring-indigo-50 transform hover:scale-105 transition-transform duration-300">
                    <img src="/img/globo.svg" alt="Logo" className="w-full h-full object-contain" />
                </div>
                <div>
                    <h1 className="text-xs font-bold leading-relaxed whitespace-pre-line text-slate-800 drop-shadow-sm">
                        {t.sidebar.title}
                    </h1>
                </div>
            </div>

            {/* Navigation */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-thin scrollbar-thumb-indigo-200">
                {/* Projects Group */}
                <div className="animate-in slide-in-from-left-4 duration-500 delay-100">
                    <div className="flex items-center gap-2 mb-3 px-2">
                        <Database size={14} className="text-indigo-400" />
                        <h2 className="text-xs font-bold uppercase tracking-wider text-indigo-900/50">{t.sidebar.datasets}</h2>
                    </div>
                    {/* Assuming structure: projects['date'][0] is the main project */}
                    {renderProjectList([
                        ...projects['2025-04-22'][0].experiments.filter(p => p.type === 'dataset'),
                        // Manually adding Stage 20 which is nested
                        projects['2025-04-22'][0].experiments[1].children[0]
                    ])}
                </div>

                {/* Experiments Group */}
                <div className="animate-in slide-in-from-left-4 duration-500 delay-200">
                    <div className="flex items-center gap-2 mb-3 px-2">
                        <Beaker size={14} className="text-indigo-400" />
                        <h2 className="text-xs font-bold uppercase tracking-wider text-indigo-900/50">{t.sidebar.experiments}</h2>
                    </div>
                    {/* Accessing deep nested experiment items */}
                    {renderProjectList([
                        ...projects['2025-04-22'][0].experiments[1].children[1].children
                    ])}
                </div>
            </div>

            {/* Footer */}
            <div className="p-4 border-t border-indigo-100 bg-white/50 backdrop-blur-sm">
                <div className="flex items-center gap-3 px-2">
                    <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></div>
                    <span className="text-xs font-medium text-slate-500">System Online</span>
                    <span className="ml-auto text-xs font-mono text-indigo-300">{t.sidebar.version}</span>
                </div>
            </div>
        </div>
    );
};

export default Sidebar;
