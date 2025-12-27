import React, { useState } from 'react';
import { Layout, FileUp, Folder, ChevronDown, ChevronRight } from 'lucide-react';

const Sidebar = () => {
    const [projects] = useState({
        '2025-08-12': [
            { id: 1, name: 'Analisis de Datos' },
            { id: 2, name: 'Prediccion de Ventas' }
        ],
        '2025-09-01': [
            { id: 3, name: 'Clasificacion de Imagenes' }
        ]
    });

    const [expandedDates, setExpandedDates] = useState(['2025-08-12']);

    const toggleDate = (date) => {
        setExpandedDates(prev =>
            prev.includes(date) ? prev.filter(d => d !== date) : [...prev, date]
        );
    };

    const fileInputRef = React.useRef(null);

    const handleFileUpload = async (event) => {
        const file = event.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/upload-template', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                alert('Plantilla cargada exitosamente');
            } else {
                alert('Error al cargar la plantilla');
            }
        } catch (error) {
            console.error('Error uploading file:', error);
            alert('Error de conexión con el servidor');
        }
    };

    const handleButtonClick = () => {
        fileInputRef.current.click();
    };

    return (
        <div className="w-64 h-screen bg-premium-gray border-r border-white/10 flex flex-col">
            <div className="p-6 border-b border-white/10">
                <h1 className="text-2xl font-serif text-premium-accent tracking-widest">TESIS</h1>
                <p className="text-xs text-gray-400 mt-1 uppercase tracking-wider">Gestor de Proyectos</p>
            </div>

            <div className="p-4">
                <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                    className="hidden"
                    accept=".docx"
                />
                <button
                    onClick={handleButtonClick}
                    className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-premium-accent to-yellow-600 text-black font-semibold py-3 px-4 rounded-lg hover:opacity-90 transition-all shadow-lg shadow-yellow-900/20"
                >
                    <FileUp size={18} />
                    <span>Cargar Plantilla</span>
                </button>
            </div>

            <div className="flex-1 overflow-y-auto px-2 py-4 space-y-2">
                {Object.entries(projects).map(([date, projectList]) => (
                    <div key={date} className="rounded-lg overflow-hidden">
                        <button
                            onClick={() => toggleDate(date)}
                            className="w-full flex items-center gap-2 p-2 text-sm text-gray-300 hover:bg-white/5 transition-colors"
                        >
                            {expandedDates.includes(date) ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                            <span className="font-mono text-premium-accent/80">{date}</span>
                        </button>

                        {expandedDates.includes(date) && (
                            <div className="ml-4 mt-1 border-l border-white/10 pl-2 space-y-1">
                                {projectList.map(project => (
                                    <button
                                        key={project.id}
                                        className="w-full text-left p-2 text-sm text-gray-400 hover:text-white hover:bg-white/5 rounded transition-all flex items-center gap-2"
                                    >
                                        <Folder size={14} />
                                        <span className="truncate">{project.name}</span>
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Sidebar;
