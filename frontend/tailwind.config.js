/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Dark blue primary (navy) palette
        navy: {
          50: "#f0f4fa",
          100: "#dde6f3",
          200: "#b9cce5",
          300: "#8babd1",
          400: "#5d86bb",
          500: "#3b6aa5",
          600: "#2a528a",
          700: "#1e3a73",
          800: "#142a5a",
          900: "#0c1d44",
          950: "#060f26",
        },
        // Soft surfaces
        paper: "#ffffff",
        canvas: "#f7f9fc",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(15,29,68,0.04), 0 8px 24px rgba(15,29,68,0.06)",
        cardLg: "0 4px 8px rgba(15,29,68,0.05), 0 24px 48px rgba(15,29,68,0.08)",
        ring: "0 0 0 4px rgba(30,58,115,0.12)",
      },
    },
  },
  plugins: [],
};
