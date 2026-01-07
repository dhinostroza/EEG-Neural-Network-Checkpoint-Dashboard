import React, { useState } from 'react';
import MainLayout from './components/Layout/MainLayout';
import Sidebar from './components/Layout/Sidebar';
import MatlabDashboard from './components/ProjectViewer/MatlabDashboard';

import { translations } from './translations';

function App() {
  const [currentProject, setCurrentProject] = useState(null);
  const [language, setLanguage] = useState('es');

  const t = translations[language];

  const renderContent = () => {
    // Project 01: Matlab EEG and its sub-datasets/experiments
    const matlabIds = [4, 8, 20, 41, 42, 78];
    if (currentProject && matlabIds.includes(currentProject.id)) {
      // Map ID to Stage if not present in object (Safety fallback)
      // Ideally Sidebar provides it, but we can enforce it here too
      const projectWithStage = { ...currentProject };
      if (!projectWithStage.stage) {
        if (projectWithStage.id === 8) projectWithStage.stage = 8;
        if (projectWithStage.id === 20) projectWithStage.stage = 20;
        if (projectWithStage.id === 41) projectWithStage.stage = 41;
        if (projectWithStage.id === 42) projectWithStage.stage = 42;
        if (projectWithStage.id === 78) projectWithStage.stage = 78;
      }
      return <MatlabDashboard project={projectWithStage} language={language} />;
    }

    return (
      <div className="flex flex-col items-center justify-center h-full text-center space-y-6 p-8">
        <div className="p-10 rounded-3xl bg-white border border-indigo-100 shadow-xl shadow-indigo-100/50 max-w-4xl w-full">
          <h2 className="text-2xl font-sans text-indigo-900 mb-6 tracking-normal font-bold leading-relaxed whitespace-pre-line">
            {t.welcome}
          </h2>
          <p className="text-slate-500 text-lg">
            {t.welcomeDesc}
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="flex min-h-screen bg-slate-100">
      <Sidebar
        onSelectProject={setCurrentProject}
        language={language}
        setLanguage={setLanguage}
      />
      <main className="flex-1 relative">
        {renderContent()}
      </main>
    </div>
  );
}

export default App;
