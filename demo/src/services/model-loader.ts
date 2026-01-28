import { CACHE_NAME, MODEL_PATHS, type ModelType, type ModelFiles } from '../config';

export type LoadMethod = 'url' | 'webfile' | 'input';

export type { ModelFiles } from '../config';

export interface LoadProgress {
  stage: 'tokenizer' | 'model';
  loaded: number;
  total: number;
  speed: number; // bytes per second
  fileName: string;
}

export type ProgressCallback = (progress: LoadProgress) => void;

/**
 * Format bytes to human readable string
 */
export function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
}

/**
 * Format speed to human readable string
 */
export function formatSpeed(bytesPerSecond: number): string {
  return `${formatBytes(bytesPerSecond)}/s`;
}

/**
 * Fetch with progress tracking and caching
 */
async function fetchWithProgress(
  url: string,
  stage: LoadProgress['stage'],
  onProgress?: ProgressCallback,
  useCache = true
): Promise<ArrayBuffer> {
  const fileName = url.split('/').pop() || url;

  // Try cache first
  if (useCache) {
    try {
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(url);
      if (cached) {
        console.log(`[Cache Hit] ${fileName}`);
        onProgress?.({
          stage,
          loaded: 1,
          total: 1,
          speed: Infinity,
          fileName: `${fileName} (cached)`,
        });
        return await cached.arrayBuffer();
      }
    } catch (e) {
      console.warn('Cache API not available:', e);
    }
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }

  const contentLength = response.headers.get('content-length');
  const total = contentLength ? parseInt(contentLength, 10) : 0;

  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onProgress?.({ stage, loaded: buffer.byteLength, total: buffer.byteLength, speed: 0, fileName });
    return buffer;
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let lastTime = Date.now();
  let lastLoaded = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    chunks.push(value);
    loaded += value.length;

    const now = Date.now();
    const elapsed = (now - lastTime) / 1000;
    if (elapsed > 0.1) {
      const speed = (loaded - lastLoaded) / elapsed;
      onProgress?.({ stage, loaded, total, speed, fileName });
      lastTime = now;
      lastLoaded = loaded;
    }
  }

  const buffer = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  // Cache the result
  if (useCache) {
    try {
      const cache = await caches.open(CACHE_NAME);
      await cache.put(url, new Response(buffer.buffer));
      console.log(`[Cached] ${fileName}`);
    } catch (e) {
      console.warn('Failed to cache:', e);
    }
  }

  return buffer.buffer;
}

/**
 * Load model from remote URLs
 */
export async function loadFromUrl(
  modelType: ModelType,
  onProgress?: ProgressCallback,
  customUrls?: { tokenizer?: string; model?: string }
): Promise<ModelFiles> {
  const paths = MODEL_PATHS[modelType];
  const tokenizerUrl = customUrls?.tokenizer || paths.tokenizer;
  const modelUrl = customUrls?.model || paths.model;

  // Load tokenizer first (smaller)
  const tokenizer = await fetchWithProgress(tokenizerUrl, 'tokenizer', onProgress);

  // Then load model (larger)
  const model = await fetchWithProgress(modelUrl, 'model', onProgress);

  return {
    tokenizer,
    model,
    tokenizerPath: tokenizerUrl,
    modelPath: modelUrl,
  };
}

/**
 * Load model using File System Access API (WebFile API)
 */
export async function loadFromWebFileAPI(
  onProgress?: ProgressCallback
): Promise<ModelFiles | null> {
  try {
    // Check if API is available
    if (!('showOpenFilePicker' in window)) {
      throw new Error('File System Access API not supported');
    }

    // Let user select files
    const fileHandles = await (window as any).showOpenFilePicker({
      multiple: true,
      types: [
        {
          description: 'Model Files',
          accept: {
            'application/octet-stream': ['.safetensors', '.json'],
          },
        },
      ],
    });

    let tokenizerBuffer: ArrayBuffer | null = null;
    let modelBuffer: ArrayBuffer | null = null;

    for (const handle of fileHandles) {
      const file: File = await handle.getFile();
      const fileName = file.name.toLowerCase();

      // Match any .json file as tokenizer
      if (fileName.endsWith('.json')) {
        onProgress?.({
          stage: 'tokenizer',
          loaded: 0,
          total: file.size,
          speed: 0,
          fileName: file.name,
        });
        tokenizerBuffer = await file.arrayBuffer();
        onProgress?.({
          stage: 'tokenizer',
          loaded: file.size,
          total: file.size,
          speed: Infinity,
          fileName: file.name,
        });
      } else if (fileName.endsWith('.safetensors')) {
        onProgress?.({
          stage: 'model',
          loaded: 0,
          total: file.size,
          speed: 0,
          fileName: file.name,
        });
        modelBuffer = await file.arrayBuffer();
        onProgress?.({
          stage: 'model',
          loaded: file.size,
          total: file.size,
          speed: Infinity,
          fileName: file.name,
        });
      }
    }

    if (!modelBuffer) {
      throw new Error('No .safetensors file selected');
    }

    if (!tokenizerBuffer) {
      throw new Error('No tokenizer.json file selected');
    }

    return {
      tokenizer: tokenizerBuffer,
      model: modelBuffer,
    };
  } catch (e) {
    if ((e as Error).name === 'AbortError') {
      return null; // User cancelled
    }
    throw e;
  }
}

/**
 * Load model from File input elements
 */
export async function loadFromFileInput(
  files: FileList,
  onProgress?: ProgressCallback
): Promise<ModelFiles | null> {
  let tokenizerBuffer: ArrayBuffer | null = null;
  let modelBuffer: ArrayBuffer | null = null;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    const fileName = file.name.toLowerCase();

    // Match any .json file as tokenizer (tokenizer.json, tokenizer_config.json, etc.)
    if (fileName.endsWith('.json')) {
      onProgress?.({
        stage: 'tokenizer',
        loaded: 0,
        total: file.size,
        speed: 0,
        fileName: file.name,
      });
      tokenizerBuffer = await file.arrayBuffer();
      onProgress?.({
        stage: 'tokenizer',
        loaded: file.size,
        total: file.size,
        speed: Infinity,
        fileName: file.name,
      });
    } else if (fileName.endsWith('.safetensors')) {
      onProgress?.({
        stage: 'model',
        loaded: 0,
        total: file.size,
        speed: 0,
        fileName: file.name,
      });
      modelBuffer = await file.arrayBuffer();
      onProgress?.({
        stage: 'model',
        loaded: file.size,
        total: file.size,
        speed: Infinity,
        fileName: file.name,
      });
    }
  }

  if (!modelBuffer) {
    throw new Error('No .safetensors file selected');
  }

  if (!tokenizerBuffer) {
    throw new Error('No tokenizer.json file selected');
  }

  return {
    tokenizer: tokenizerBuffer,
    model: modelBuffer,
  };
}

/**
 * Clear model cache
 */
export async function clearModelCache(): Promise<void> {
  try {
    await caches.delete(CACHE_NAME);
    console.log('Model cache cleared');
  } catch (e) {
    console.warn('Failed to clear cache:', e);
  }
}

/**
 * Get cached model list
 */
export async function getCachedModels(): Promise<string[]> {
  try {
    const cache = await caches.open(CACHE_NAME);
    const keys = await cache.keys();
    return keys.map((req) => req.url);
  } catch (e) {
    console.warn('Failed to get cached models:', e);
    return [];
  }
}
