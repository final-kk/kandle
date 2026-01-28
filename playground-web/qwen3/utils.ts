/**
 * Shared utilities for Qwen3 Web Playground
 */

import { Tensor } from '@kandle/core';
import { type Qwen3ModelConfig } from '@kandle/model-utils';

// ============================================================================
// Constants
// ============================================================================

// Models will be loaded from file inputs, but we define config here.
// Use 'mps' equivalent if available or fallback to reasonable defaults ?
// Actually we just use it for config.

// Qwen3-0.6B Config
export const QWEN3_CONFIG: Qwen3ModelConfig = {
    vocabSize: 151936,
    hiddenSize: 1024,
    intermediateSize: 3072,
    numHiddenLayers: 28,
    numAttentionHeads: 16,
    numKeyValueHeads: 8,
    headDim: 128,
    maxPositionEmbeddings: 40960,
    ropeTheta: 1000000,
    rmsNormEps: 1e-6,
    attentionBias: false,
    mlpBias: false,
    dtype: 'float32',
};

export const EOS_TOKEN_IDS = [151645, 151643];

// ============================================================================
// Logger
// ============================================================================

export const logger = {
    log: (msg: string) => appendLog(msg, 'log'),
    logGroup: (msg: string) => appendLog(`\n=== ${msg} ===`, 'group'),
    info: (msg: string) => appendLog(msg, 'info'),
    success: (msg: string) => appendLog(msg, 'success'),
    error: (msg: string) => appendLog(msg, 'error'),
    warn: (msg: string) => appendLog(msg, 'warn'),
};

function appendLog(msg: string, type: string) {
    const outputEl = document.getElementById('output');
    if (outputEl) {
        const div = document.createElement('div');
        div.textContent = msg;
        div.className = `log-${type}`;
        outputEl.appendChild(div);
        outputEl.scrollTop = outputEl.scrollHeight;
    }
    console.log(`[${type}]`, msg);
}

// ============================================================================
// Utilities
// ============================================================================

export function assert(condition: boolean, msg: string): void {
    if (!condition) {
        logger.error(`ASSERTION FAILED: ${msg}`);
        throw new Error(msg);
    }
}

export async function getTensorStats(tensor: Tensor): Promise<{
    min: number;
    max: number;
    mean: number;
    isFinite: boolean;
}> {
    const data = await tensor.dataAsync();
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let allFinite = true;

    for (let i = 0; i < data.length; i++) {
        const val = Number(data[i]);
        if (!Number.isFinite(val)) {
            allFinite = false;
        }
        if (val < min) min = val;
        if (val > max) max = val;
        sum += val;
    }

    return {
        min,
        max,
        mean: sum / data.length,
        isFinite: allFinite,
    };
}
