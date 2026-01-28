import { defineConfig } from 'vite';

export default defineConfig({
    build: {
        target: 'esnext',
    },
    optimizeDeps: {
        exclude: ['@kandle/backend-webgpu'], // if needed
    }
});
