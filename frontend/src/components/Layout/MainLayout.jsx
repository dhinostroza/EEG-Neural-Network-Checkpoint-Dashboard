import React from 'react';
import Sidebar from './Sidebar';

const MainLayout = ({ children }) => {
    return (
        <div className="flex min-h-screen bg-premium-dark text-gray-200">
            <Sidebar />
            <main className="flex-1 overflow-hidden relative">
                <div className="absolute inset-0 bg-gradient-to-br from-premium-dark via-[#0F0F0F] to-[#1a1a1a] z-0 pointer-events-none" />
                <div className="relative z-10 w-full h-full p-8 overflow-y-auto">
                    {children}
                </div>
            </main>
        </div>
    );
};

export default MainLayout;
