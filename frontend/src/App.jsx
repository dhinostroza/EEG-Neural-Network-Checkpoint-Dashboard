import React from 'react';
import MainLayout from './components/Layout/MainLayout';

function App() {
  return (
    <MainLayout>
      <div className="flex flex-col items-center justify-center h-full text-center space-y-6">
        <div className="p-8 rounded-full bg-premium-gray border border-white/5 shadow-2xl">
          <h2 className="text-3xl font-serif text-premium-accent mb-2">Bienvenido a Tesis</h2>
          <p className="text-gray-400 max-w-md">
            Selecciona un proyecto del menú lateral para ver su información técnica y generar documentos.
          </p>
        </div>
      </div>
    </MainLayout>
  );
}

export default App;
