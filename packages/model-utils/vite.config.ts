import { defineConfig } from 'vite';
import { resolve } from 'path';
import dts from 'vite-plugin-dts';

export default defineConfig({
    build: {
        target: 'esnext',
        lib: {
            entry: resolve(__dirname, 'src/index.ts'),
            name: 'KandleModelUtils',
            formats: ['es', 'cjs'],
            fileName: (format) => `index.${format === 'es' ? 'js' : 'cjs'}`
        },
        sourcemap: true,
        minify: false,
        rollupOptions: {
            external: [
                '@kandle/core',
                '@kandle/types',
                '@kandle/utils',
            ],
            output: {
                globals: {
                    '@kandle/core': 'KandleCore',
                    '@kandle/utils': 'KandleUtils',
                    '@kandle/types': 'KandleTypes',
                }
            }
        },
        outDir: "dist",
        emptyOutDir: true,
    },
    plugins: [
        dts({
            insertTypesEntry: true,
            rollupTypes: false,
        })
    ]
});
