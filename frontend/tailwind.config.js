/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0a0f1a",
        card: "#141b2d",
        idbi: "#004B87",
        hot: "#f97316",
        warm: "#eab308",
        cold: "#64748b",
        converted: "#10b981"
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
