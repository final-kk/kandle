// Model and resource paths configuration

export type WhisperModelSize = 'tiny' | 'base' | 'small' | 'medium' | 'large-v3';

export const WHISPER_MODEL_URLS: Record<WhisperModelSize, { tokenizer: string; model: string }> = {
  tiny: {
    tokenizer: 'https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json',
    model: 'https://huggingface.co/openai/whisper-tiny/resolve/main/model.safetensors',
  },
  base: {
    tokenizer: 'https://huggingface.co/openai/whisper-base/resolve/main/tokenizer.json',
    model: 'https://huggingface.co/openai/whisper-base/resolve/main/model.safetensors',
  },
  small: {
    tokenizer: 'https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json',
    model: 'https://huggingface.co/openai/whisper-small/resolve/main/model.safetensors',
  },
  medium: {
    tokenizer: 'https://huggingface.co/openai/whisper-medium/resolve/main/tokenizer.json',
    model: 'https://huggingface.co/openai/whisper-medium/resolve/main/model.safetensors',
  },
  'large-v3': {
    tokenizer: 'https://huggingface.co/openai/whisper-large-v3/resolve/main/tokenizer.json',
    model: 'https://huggingface.co/openai/whisper-large-v3/resolve/main/model.safetensors',
  },
};

export const QWEN3_MODEL_URLS = {
  '0.6b': {
    tokenizer: 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/tokenizer.json',
    model: 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors',
  },
} as const;

export type Qwen3ModelSize = keyof typeof QWEN3_MODEL_URLS;

export const MODEL_PATHS = {
  qwen3: QWEN3_MODEL_URLS['0.6b'],
  whisper: WHISPER_MODEL_URLS['tiny'],
} as const;

export const CACHE_NAME = 'kandle-models-v1';

export type ModelType = 'qwen3' | 'whisper';

export interface ModelFiles {
  tokenizer: ArrayBuffer | null;
  model: ArrayBuffer;
  tokenizerPath?: string;
  modelPath?: string;
}
