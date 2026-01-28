/// <reference types="vite/client" />

declare module '@huggingface/tokenizers' {
  export class Tokenizer {
    constructor(json: Record<string, unknown>, options?: Record<string, unknown>);
    encode(text: string, addSpecialTokens?: boolean): Promise<{ ids: number[] }>;
    decode(ids: number[], skipSpecialTokens?: boolean): Promise<string>;
  }
}
