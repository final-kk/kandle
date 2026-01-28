import { defineConfig } from 'vite';
import { resolve } from 'path';
import dts from 'vite-plugin-dts';

export default defineConfig({
    build: {
        target: 'esnext',
        lib: {
            entry: resolve(__dirname, 'src/index.ts'),
            name: 'KandleCore',
            formats: ['es', 'cjs'],
            fileName: (format) => `index.${format === 'es' ? 'js' : 'cjs'}`
        },
        sourcemap: true,
        minify: false,
        rollupOptions: {
            external: [
                '@kandle/types',
                '@kandle/utils',
            ],
            output: {
                globals: {
                    '@kandle/utils': 'KandleUtils',
                    '@kandle/types': 'KandleTypes',
                }
            }
        },
        outDir: "dist",
        emptyOutDir: true,
    },
    esbuild: {
        target: 'esnext'
    },
    plugins: [
        dts({
            insertTypesEntry: true,
            rollupTypes: false,
        })
    ],
    root: '.',
    server: {
        port: 8173,
        host: true,
        fs: {
            allow: ['..', '../..'],
        }
    }
});
