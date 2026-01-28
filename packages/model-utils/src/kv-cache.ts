/**
 * KV Cache - 用于 Transformer 推理的 Key-Value 缓存
 *
 * 参考 HuggingFace Transformers 的 StaticCache 设计：
 * - 预分配固定大小的缓存，避免动态内存分配
 * - 配合 Memory Pool 实现高效的缓存复用
 * - 支持增量 decode 场景
 *
 * @example
 * ```ts
 * // 创建 KV Cache
 * const cache = new StaticKVCache({
 *     numLayers: 28,
 *     numKvHeads: 8,
 *     headDim: 128,
 *     maxSeqLen: 2048,
 * });
 *
 * // Prefill 阶段: 处理完整 prompt
 * const [k, v] = cache.update(layerIdx, keyStates, valueStates, 0);
 *
 * // Decode 阶段: 每次处理一个新 token
 * const [k, v] = cache.update(layerIdx, keyStates, valueStates, seenTokens);
 * ```
 *
 * @module @kandle/model-utils/kv-cache
 */

import type { DType } from "@kandle/types";
import { Tensor, zeros, slice, contiguous, internal, keep } from "@kandle/core";

// ============================================================================
// Types
// ============================================================================

/**
 * StaticKVCache 构造选项
 */
export interface StaticKVCacheOptions {
    /** Transformer 层数 */
    numLayers: number;

    /** KV 头数量 (用于 GQA) */
    numKvHeads: number;

    /** 每个头的维度 */
    headDim: number;

    /** 最大序列长度 */
    maxSeqLen: number;

    /** Batch size (默认 1) */
    batchSize?: number;

    /** 数据类型 (默认 float32) */
    dtype?: DType;
}

/**
 * 单层的 K/V 缓存对
 */
export interface LayerKVCache {
    /** Key 缓存，形状 (batch, numKvHeads, maxSeqLen, headDim) */
    keyCache: Tensor;

    /** Value 缓存，形状 (batch, numKvHeads, maxSeqLen, headDim) */
    valueCache: Tensor;
}

// ============================================================================
// StaticKVCache Class
// ============================================================================

/**
 * StaticKVCache - 静态 KV 缓存
 *
 * 特点：
 * - 预分配固定大小的缓存，避免推理时动态内存分配
 * - 使用 slice + copy_ 组合写入新的 K/V，使用 slice 读取历史 K/V
 * - 每层独立管理，支持不同层的独立更新
 *
 * 缓存布局 (BHSD - Batch, Heads, Sequence, Dimension):
 * - keyCache[layer]: (batch, numKvHeads, maxSeqLen, headDim)
 * - valueCache[layer]: (batch, numKvHeads, maxSeqLen, headDim)
 *
 * 使用流程:
 * 1. 创建 cache
 * 2. Prefill: update(layer, k, v, 0) - 写入 prompt 的 K/V
 * 3. Decode: update(layer, k, v, pos) - 每次 decode 写入新 token 的 K/V
 * 4. update() 返回到目前为止的有效缓存切片
 */
export class StaticKVCache {
    readonly numLayers: number;
    readonly numKvHeads: number;
    readonly headDim: number;
    readonly maxSeqLen: number;
    readonly batchSize: number;
    readonly dtype: DType;

    /** 当前已填充的序列长度 */
    private _seenTokens: number = 0;

    /** 每层的 K/V 缓存 */
    private _caches: LayerKVCache[];

    constructor(options: StaticKVCacheOptions) {
        const {
            numLayers,
            numKvHeads,
            headDim,
            maxSeqLen,
            batchSize = 1,
            dtype = "float32",
        } = options;

        this.numLayers = numLayers;
        this.numKvHeads = numKvHeads;
        this.headDim = headDim;
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
        this.dtype = dtype;

        // 预分配所有层的缓存
        this._caches = [];
        for (let i = 0; i < numLayers; i++) {
            this._caches.push({
                keyCache: keep(zeros([batchSize, numKvHeads, maxSeqLen, headDim], dtype)),
                valueCache: keep(zeros([batchSize, numKvHeads, maxSeqLen, headDim], dtype)),
            });
        }
    }

    /**
     * 获取当前已处理的 token 数量
     */
    get seenTokens(): number {
        return this._seenTokens;
    }

    /**
     * 获取指定层的缓存
     *
     * @param layerIdx 层索引
     * @returns 该层的 K/V 缓存
     */
    getLayerCache(layerIdx: number): LayerKVCache {
        if (layerIdx < 0 || layerIdx >= this.numLayers) {
            throw new Error(`Layer index ${layerIdx} out of range [0, ${this.numLayers})`);
        }
        return this._caches[layerIdx];
    }

    /**
     * 更新指定层的 KV 缓存
     *
     * 将新的 key/value 写入缓存的指定位置
     *
     * @param layerIdx 层索引
     * @param key 新的 key，形状 (batch, numKvHeads, newSeqLen, headDim)
     * @param value 新的 value，形状 (batch, numKvHeads, newSeqLen, headDim)
     * @param startPos 写入的起始位置
     * @returns 更新后的完整缓存 [keyCache, valueCache]，切片到有效范围
     */
    update(layerIdx: number, key: Tensor, value: Tensor, startPos: number): [Tensor, Tensor] {
        const cache = this.getLayerCache(layerIdx);
        const newSeqLen = key.shape[2];
        const endPos = startPos + newSeqLen;

        if (endPos > this.maxSeqLen) {
            throw new Error(
                `Cache overflow: trying to write to position ${startPos}-${endPos}, ` +
                    `but maxSeqLen is ${this.maxSeqLen}`
            );
        }

        // 写入新的 K/V 到缓存
        // 使用 slice + copy_ 组合方案 (PyTorch 标准方式)
        // 1. 创建目标位置的 slice view
        // 2. 使用 copy_ 原地写入
        const writeSliceStr = `:, :, ${startPos}:${endPos}, :`;

        const keyView = slice(cache.keyCache, writeSliceStr);
        internal.copy_(keyView._handle, key._handle);
        keyView.dispose(); // view 共享 storage，dispose 只减少引用计数

        const valueView = slice(cache.valueCache, writeSliceStr);
        internal.copy_(valueView._handle, value._handle);
        valueView.dispose();

        // 更新 seen tokens (仅在写入新位置时更新)
        if (endPos > this._seenTokens) {
            this._seenTokens = endPos;
        }

        // 返回到目前为止的有效缓存切片
        // 使用 Python 风格切片语法: ":, :, :endPos, :"
        const sliceStr = `:, :, :${endPos}, :`;
        const keySlice = slice(cache.keyCache, sliceStr);
        const valueSlice = slice(cache.valueCache, sliceStr);

        // 必须返回 contiguous 数据
        // slice view 共享底层 storage，调用方 dispose 会导致 cache storage 被释放
        // const keyContiguous = contiguous(keySlice);
        // const valueContiguous = contiguous(valueSlice);

        // 释放 slice view (它们共享底层缓存的 storage，dispose 只减少引用计数)
        // keySlice.dispose();
        // valueSlice.dispose();

        // return [keyContiguous, valueContiguous];
        return [keySlice, valueSlice];
    }

    /**
     * 获取到指定位置的有效 K/V 缓存
     *
     * @param layerIdx 层索引
     * @param upToPos 读取到的位置 (不包含)，默认为 seenTokens
     * @returns [keyCache, valueCache] 切片
     */
    getKeyValue(layerIdx: number, upToPos?: number): [Tensor, Tensor] {
        const cache = this.getLayerCache(layerIdx);
        const end = upToPos ?? this._seenTokens;

        if (end <= 0) {
            throw new Error("No tokens in cache yet");
        }

        const sliceStr = `:, :, :${end}, :`;
        // 必须返回 contiguous 数据（slice view 共享 storage，调用方 dispose 会导致问题）
        return [
            contiguous(slice(cache.keyCache, sliceStr)),
            contiguous(slice(cache.valueCache, sliceStr)),
        ];
    }

    /**
     * 重置缓存状态
     *
     * 清零已处理 token 计数，缓存内容保留（下次写入会覆盖）
     */
    reset(): void {
        this._seenTokens = 0;
    }

    /**
     * 回滚到指定位置
     *
     * 用于实现"后退"功能：将缓存状态回滚到之前的某个位置。
     * 由于 KV Cache 使用覆盖写入机制，只需修改计数器即可。
     * 下次写入会从目标位置开始覆盖旧数据。
     *
     * @param targetPosition 目标位置（必须 <= 当前 seenTokens）
     * @throws 如果目标位置无效
     *
     * @example
     * ```ts
     * // 用户生成了 5 个 token 后想回退 1 步
     * const promptLen = 10;
     * const generated = 5;
     * cache.rollback(promptLen + generated - 1);  // 回到第 4 个 token 的状态
     * ```
     */
    rollback(targetPosition: number): void {
        if (targetPosition < 0) {
            throw new Error(`Invalid rollback position: ${targetPosition} (must be >= 0)`);
        }
        if (targetPosition > this._seenTokens) {
            throw new Error(
                `Cannot rollback forward: target ${targetPosition} > current ${this._seenTokens}`
            );
        }
        this._seenTokens = targetPosition;
    }

    /**
     * 释放所有缓存占用的显存
     */
    dispose(): void {
        for (const cache of this._caches) {
            // 调用底层 dispose（如果可用）
            if ("dispose" in cache.keyCache._handle) {
                (cache.keyCache._handle as any).dispose();
            }
            if ("dispose" in cache.valueCache._handle) {
                (cache.valueCache._handle as any).dispose();
            }
        }
        this._caches = [];
        this._seenTokens = 0;
    }

    /**
     * 获取缓存占用的总字节数
     */
    get totalBytes(): number {
        const bytesPerElement = this.dtype === "float32" ? 4 : 2;
        const elementsPerCache = this.batchSize * this.numKvHeads * this.maxSeqLen * this.headDim;
        // K + V for each layer
        return 2 * this.numLayers * elementsPerCache * bytesPerElement;
    }

    /**
     * 格式化内存大小
     */
    get totalMemoryMB(): string {
        return (this.totalBytes / (1024 * 1024)).toFixed(2);
    }
}
