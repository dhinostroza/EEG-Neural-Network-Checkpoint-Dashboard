/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            fontFamily: {
                serif: ['"Trajan Pro"', 'serif'],
                sans: ['Inter', 'sans-serif'],
            },
            colors: {
                'premium-dark': '#0a0a0a',
                'premium-gray': '#1a1a1a',
                'premium-accent': '#d4af37', // Gold-ish for "Premium" feel
                'glass-white': 'rgba(255, 255, 255, 0.05)',
            }
        },
    },
    plugins: [],
}
