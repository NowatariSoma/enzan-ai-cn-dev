import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: ['class'],
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './features/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        xs: '0.8125rem', // 13px base from @front
        sm: '0.875rem',
        base: '0.9375rem', // 15px
        lg: '1rem',
        xl: '1.125rem',
        '2xl': '1.25rem',
        '3xl': '1.5rem',
      },
      screens: {
        'mobile': '768px', // @front mobile viewport
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic':
          'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'noisy': 'url("data:image/svg+xml,%3Csvg viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg"%3E%3Cfilter id="noiseFilter"%3E%3CfeTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="4" stitchTiles="stitch"/%3E%3C/filter%3E%3Crect width="100%25" height="100%25" filter="url(%23noiseFilter)" opacity="0.02"/%3E%3C/svg%3E")',
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      colors: {
        // @front inspired color palette
        gray: {
          0: '#ffffff',
          10: '#fcfcfc',
          15: '#f5f5f5',
          20: '#ebebeb',
          25: '#d6d6d6',
          30: '#cccccc',
          35: '#b3b3b3',
          40: '#999999',
          45: '#666666',
          50: '#4c4c4c',
          55: '#333333',
          60: '#292929',
          65: '#222222',
          70: '#1d1d1d',
          75: '#1b1b1b',
          80: '#171717',
          85: '#141414',
          90: '#0f0f0f',
          100: '#000000',
        },
        blue: {
          DEFAULT: '#1961ed',
          10: '#e4e4fd',
          20: '#c5c5fb',
          30: '#9797f0',
          40: '#5a5ac0',
          50: '#2d2e6d',
          60: '#22244f',
          70: '#1e203f',
          80: '#1a1d2d',
        },
        green: {
          DEFAULT: '#55ef3c',
          10: '#f3fde9',
          20: '#e0fbd1',
          30: '#b9f5a3',
          40: '#7edc6a',
          50: '#3f7d2e',
          60: '#2a5822',
          70: '#23421e',
          80: '#1d2d1b',
        },
        red: {
          DEFAULT: '#f83e3e',
          10: '#fde4e4',
          20: '#fbc5c5',
          30: '#f09797',
          40: '#bf5a5a',
          50: '#6d2d2d',
          60: '#502222',
          70: '#3f1e1e',
          80: '#2d1a1a',
        },
        yellow: {
          DEFAULT: '#ffd338',
          10: '#fffbeb',
          20: '#fff6d7',
          30: '#ffedaf',
          40: '#ffe074',
          50: '#b99b2e',
          60: '#746224',
          70: '#453d1e',
          80: '#2e2a1a',
        },
        // shadcn/ui colors
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        chart: {
          '1': 'hsl(var(--chart-1))',
          '2': 'hsl(var(--chart-2))',
          '3': 'hsl(var(--chart-3))',
          '4': 'hsl(var(--chart-4))',
          '5': 'hsl(var(--chart-5))',
        },
      },
      boxShadow: {
        'soft': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
        'medium': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'strong': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'extraLight': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      keyframes: {
        'accordion-down': {
          from: {
            height: '0',
          },
          to: {
            height: 'var(--radix-accordion-content-height)',
          },
        },
        'accordion-up': {
          from: {
            height: 'var(--radix-accordion-content-height)',
          },
          to: {
            height: '0',
          },
        },
        'fade-in': {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in': {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(0)' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'fade-in': 'fade-in 0.3s ease-out',
        'slide-in': 'slide-in 0.3s ease-out',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};
export default config;
