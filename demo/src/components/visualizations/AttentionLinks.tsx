/**
 * AttentionLinks - Attention 权重可视化组件
 *
 * 显示当前 token 对之前所有 token 的 attention 权重分布。
 * 使用 SVG 绘制连线，线条粗细和颜色反映 attention 强度。
 *
 * 支持：
 * - 多层/多头选择（包含 Average 选项）
 * - 热力图模式
 * - 连线模式（显示当前 token 对历史 token 的关注）
 */

import { useState, useMemo, useRef, useEffect, useCallback, Fragment } from "react";
import type { LayerAttentionData } from "../../types";

// ============================================================================
// Types
// ============================================================================

/** 旧的简化数据格式（用于兼容） */
export interface AttentionData {
    /** attention 权重矩阵 [numHeads, seqLen, seqLen] 或 [seqLen, seqLen] */
    weights: number[][][] | number[][];
    /** 层索引 */
    layerIndex: number;
    /** token 文本列表 */
    tokens: string[];
}

/** 头选择值："average" 或数字索引 */
export type HeadSelection = "average" | number;

export interface AttentionLinksProps {
    /** Attention 数据（支持新格式 LayerAttentionData[] 或旧格式 AttentionData） */
    data: LayerAttentionData[] | AttentionData | null;

    /** Token 文本列表（新格式需要） */
    tokens?: string[];

    /** 可用的层数 */
    numLayers?: number;

    /** 当前选中的层 */
    selectedLayer?: number;

    /** 层切换回调 */
    onLayerChange?: (layer: number) => void;

    /** 当前选中的头 */
    selectedHead?: HeadSelection;

    /** 头切换回调 */
    onHeadChange?: (head: HeadSelection) => void;

    /** 显示模式 */
    mode?: "heatmap" | "links";

    /** 最大高度 */
    maxHeight?: number;

    /** 是否显示数值 */
    showValues?: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

/** 检查是否为新格式 LayerAttentionData[] */
function isLayerAttentionDataArray(
    data: LayerAttentionData[] | AttentionData | null
): data is LayerAttentionData[] {
    if (!data || !Array.isArray(data)) return false;
    if (data.length === 0) return false;
    // LayerAttentionData 有 layerIndex, numHeads, querySeqLen, keySeqLen, weights
    const first = data[0] as unknown as Record<string, unknown>;
    // 注意: Float32Array 通过 Worker Transferable 传输后, instanceof 可能失败
    // 使用 ArrayBuffer.isView 或检查 constructor.name 作为备选
    const hasValidWeights =
        first.weights instanceof Float32Array ||
        (first.weights && ArrayBuffer.isView(first.weights as ArrayBufferView)) ||
        (first.weights && (first.weights as { constructor?: { name?: string } }).constructor?.name === "Float32Array");

    return (
        typeof first.layerIndex === "number" &&
        typeof first.numHeads === "number" &&
        !!hasValidWeights
    );
}

/** 提取单个头的 attention */
function extractHeadAttention(
    weights: Float32Array,
    headIdx: number,
    querySeqLen: number,
    keySeqLen: number
): number[][] {
    const result: number[][] = [];
    const offset = headIdx * querySeqLen * keySeqLen;

    for (let q = 0; q < querySeqLen; q++) {
        const row: number[] = [];
        for (let k = 0; k < keySeqLen; k++) {
            row.push(weights[offset + q * keySeqLen + k]);
        }
        result.push(row);
    }

    return result;
}

/**
 * 从历史中提取指定 head 或计算所有 head 的平均
 * 用于 Main Component 中的数据处理
 */
interface LayerAttentionHistoryData {
    numHeads: number;
    weights: number[][][]; // [numHeads][seqLen][seqLen]
}

function extractOrAverageFromHistory(
    history: LayerAttentionHistoryData,
    selectedHead: HeadSelection
): number[][] {
    const { numHeads, weights } = history;

    if (selectedHead === "average") {
        // 计算所有 head 的平均
        if (weights.length === 0 || weights[0].length === 0) {
            return [];
        }
        const seqLen = weights[0].length;
        const keyLen = weights[0][0]?.length || 0;

        const result: number[][] = Array.from({ length: seqLen }, () =>
            Array(keyLen).fill(0)
        );

        for (let h = 0; h < numHeads; h++) {
            for (let q = 0; q < seqLen; q++) {
                for (let k = 0; k < keyLen; k++) {
                    result[q][k] += (weights[h]?.[q]?.[k] || 0) / numHeads;
                }
            }
        }
        return result;
    } else {
        // 返回指定 head 的权重
        const headIdx = Math.min(selectedHead, numHeads - 1);
        return weights[headIdx] || [];
    }
}

/** 将 attention 权重映射到颜色 */
function weightToColor(weight: number, alpha: number = 1): string {
    // 使用 cyan -> purple 渐变
    const intensity = Math.min(1, Math.max(0, weight));
    const r = Math.round(intensity * 168); // purple
    const g = Math.round((1 - intensity) * 255 * 0.8); // cyan component
    const b = Math.round(255 - intensity * 87);
    return `rgba(${r}, ${g}, ${b}, ${alpha * intensity + 0.1})`;
}

/** 将 attention 权重映射到线宽 */
function weightToLineWidth(weight: number): number {
    return 1 + weight * 5;
}

// ============================================================================
// Heatmap Sub-component (Exported)
// ============================================================================

export interface HeatmapViewProps {
    weights: number[][];
    tokens: string[];
    showValues?: boolean;
    /** 可选的标题 */
    title?: string;
}

export function HeatmapView({ weights, tokens, showValues = true, title }: HeatmapViewProps) {
    const seqLen = tokens.length;
    const cellSize = Math.min(40, Math.max(20, 400 / seqLen));

    return (
        <div>
            {title && (
                <div className="text-xs text-gray-400 px-2 py-1 border-b border-gray-800">
                    {title}
                </div>
            )}
            <div
                className="grid gap-px bg-gray-800 w-fit"
                style={{
                    gridTemplateColumns: `80px repeat(${seqLen}, ${cellSize}px)`,
                }}
            >
                {/* Header row - query tokens */}
                <div className="bg-gray-900 p-1 text-xs text-gray-500 text-center">Q \ K</div>
                {tokens.map((token, i) => (
                    <div
                        key={`h-${i}`}
                        className="bg-gray-900 p-1 text-xs text-gray-400 text-center truncate"
                        title={token}
                        style={{ width: cellSize, height: 24 }}
                    >
                        {token.slice(0, 3)}
                    </div>
                ))}

                {/* Data rows */}
                {weights.map((row, qi) => (
                    <Fragment key={`row-${qi}`}>
                        {/* Row label - key token */}
                        <div
                            className="bg-gray-900 p-1 text-xs text-gray-400 truncate flex items-center"
                            title={tokens[qi]}
                        >
                            {tokens[qi]?.slice(0, 8) ?? `[${qi}]`}
                        </div>
                        {/* Attention cells */}
                        {row.map((weight, ki) => (
                            <div
                                key={`c-${qi}-${ki}`}
                                className="flex items-center justify-center text-xs font-mono transition-all hover:ring-1 hover:ring-cyber-cyan"
                                style={{
                                    backgroundColor: weightToColor(weight, 0.9),
                                    width: cellSize,
                                    height: cellSize,
                                }}
                                title={`${tokens[qi] ?? qi} -> ${tokens[ki] ?? ki}: ${(weight * 100).toFixed(1)}%`}
                            >
                                {showValues && weight > 0.05 && (
                                    <span className="text-white/80 text-[10px]">
                                        {(weight * 100).toFixed(0)}
                                    </span>
                                )}
                            </div>
                        ))}
                    </Fragment>
                ))}
            </div>
        </div>
    );
}

// ============================================================================
// Links Sub-component
// ============================================================================

interface LinksViewProps {
    weights: number[][];
    tokens: string[];
}

function LinksView({ weights, tokens }: LinksViewProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [tokenPositions, setTokenPositions] = useState<{ x: number; y: number }[]>([]);

    // weights 形状: [querySeqLen, keySeqLen]
    // - Prefill: querySeqLen = keySeqLen = promptLength
    // - Decode: querySeqLen = 1, keySeqLen = totalSeqLen

    // 我们想要显示"当前 token"对所有之前 token 的 attention
    // 所以我们总是取 weights 的最后一行
    const querySeqLen = weights.length;
    const keySeqLen = weights[0]?.length || 0;

    // 当前 token 的 attention 权重（最后一行）
    const currentWeights = weights[querySeqLen - 1] || [];

    // 当前 token 在完整序列中的位置 = keySeqLen - 1
    // （因为 decode 阶段的 weights[0] 对应的是最后一个 query token）
    const currentIndex = keySeqLen - 1;

    // 计算 token 位置
    useEffect(() => {
        if (!containerRef.current) return;

        const container = containerRef.current;
        const tokenElements = container.querySelectorAll("[data-token-idx]");
        const positions: { x: number; y: number }[] = [];

        tokenElements.forEach((el) => {
            const rect = el.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            positions.push({
                x: rect.left - containerRect.left + rect.width / 2,
                y: rect.top - containerRect.top + rect.height / 2,
            });
        });

        setTokenPositions(positions);
    }, [tokens]);

    // 筛选出有意义的连线（权重 > 5%）
    const significantLinks = useMemo(() => {
        return currentWeights
            .map((weight, idx) => ({ idx, weight }))
            .filter((l) => l.weight > 0.05 && l.idx !== currentIndex)
            .sort((a, b) => b.weight - a.weight)
            .slice(0, 10); // 最多显示10条
    }, [currentWeights, currentIndex]);

    return (
        <div ref={containerRef} className="relative min-h-[200px]">
            {/* SVG for links */}
            <svg
                className="absolute inset-0 pointer-events-none"
                style={{ width: "100%", height: "100%" }}
            >
                {tokenPositions.length > 0 &&
                    significantLinks.map(({ idx, weight }) => {
                        const from = tokenPositions[idx];
                        const to = tokenPositions[currentIndex];
                        if (!from || !to) return null;

                        return (
                            <line
                                key={idx}
                                x1={from.x}
                                y1={from.y}
                                x2={to.x}
                                y2={to.y}
                                stroke={weightToColor(weight)}
                                strokeWidth={weightToLineWidth(weight)}
                                strokeLinecap="round"
                                opacity={0.7}
                            />
                        );
                    })}
            </svg>

            {/* Token chips */}
            <div className="flex flex-wrap gap-2 p-4 relative z-10">
                {tokens.map((token, idx) => {
                    const isCurrent = idx === currentIndex;
                    const weight = currentWeights[idx] || 0;
                    const isSignificant = weight > 0.1;

                    return (
                        <span
                            key={idx}
                            data-token-idx={idx}
                            className={`
                                px-2 py-1 rounded text-xs font-mono transition-all
                                ${isCurrent
                                    ? "bg-cyber-purple text-white ring-2 ring-cyber-purple/50"
                                    : isSignificant
                                        ? "bg-cyber-cyan/30 text-cyber-cyan"
                                        : "bg-gray-800 text-gray-400"
                                }
                            `}
                            title={
                                isCurrent
                                    ? `Current token: ${token}`
                                    : `${token}: ${(weight * 100).toFixed(1)}% attention`
                            }
                        >
                            {token}
                            {isSignificant && !isCurrent && (
                                <span className="ml-1 text-[10px] opacity-70">
                                    {(weight * 100).toFixed(0)}%
                                </span>
                            )}
                        </span>
                    );
                })}
            </div>

            {/* Legend */}
            <div className="flex items-center gap-4 px-4 py-2 text-xs text-gray-500 border-t border-gray-800">
                <span className="flex items-center gap-1">
                    <span className="w-3 h-3 rounded bg-cyber-purple" />
                    Current token
                </span>
                <span className="flex items-center gap-1">
                    <span className="w-3 h-3 rounded bg-cyber-cyan/50" />
                    High attention
                </span>
                <span>Line width = attention strength</span>
            </div>
        </div>
    );
}

// ============================================================================
// Main Component
// ============================================================================

/**
 * 累积的 attention 历史，按层存储原始 per-head 数据
 * 结构: layerIndex -> { lastKeySeqLen, weights: [numHeads][seqLen][seqLen] }
 */
interface LayerAttentionHistory {
    lastKeySeqLen: number;
    numHeads: number;
    /** 每个 head 的完整 attention 矩阵 [numHeads][seqLen][seqLen] */
    weights: number[][][];
}

export function AttentionLinks({
    data,
    tokens: externalTokens,
    numLayers = 28,
    selectedLayer = 0,
    onLayerChange,
    selectedHead: externalSelectedHead,
    onHeadChange,
    mode = "links",
    maxHeight = 400,
    showValues = true,
}: AttentionLinksProps) {
    const [viewMode, setViewMode] = useState<"heatmap" | "links">(mode);
    // 内部状态（当外部不控制时使用）
    const [internalSelectedHead, setInternalSelectedHead] = useState<HeadSelection>(0);

    // 累积历史 attention 数据（用于 decode 阶段构建完整矩阵）
    // key: layerIndex, value: 该层所有 head 的累积历史
    const attentionHistoryRef = useRef<Map<number, LayerAttentionHistory>>(new Map());

    // 使用外部或内部的头选择
    const selectedHead = externalSelectedHead ?? internalSelectedHead;
    const handleHeadChange = useCallback(
        (head: HeadSelection) => {
            if (onHeadChange) {
                onHeadChange(head);
            } else {
                setInternalSelectedHead(head);
            }
        },
        [onHeadChange]
    );

    // 处理 attention 数据 - 支持新旧两种格式，并累积历史
    const processedData = useMemo(() => {
        if (!data) return null;

        // 新格式: LayerAttentionData[]
        if (isLayerAttentionDataArray(data)) {
            // 找到选中的层
            const layerData = data.find((d) => d.layerIndex === selectedLayer);
            if (!layerData) {
                return null;
            }

            const { weights, numHeads, querySeqLen, keySeqLen } = layerData;
            const history = attentionHistoryRef.current.get(selectedLayer);

            // 检测是否需要重置历史（新的生成开始，keySeqLen 变小了）
            if (history && keySeqLen < history.lastKeySeqLen) {
                attentionHistoryRef.current.clear();
            }

            // 检测是否已经处理过这个 keySeqLen（避免 useMemo 多次执行导致重复累加）
            if (history && history.lastKeySeqLen === keySeqLen) {
                // 直接从缓存返回
                const cachedWeights = extractOrAverageFromHistory(history, selectedHead);
                return {
                    weights: cachedWeights,
                    numHeads,
                    tokens: externalTokens || [],
                    availableLayers: data.map((d) => d.layerIndex),
                };
            }

            // 构建或更新历史
            let updatedHistory: LayerAttentionHistory;

            if (querySeqLen === keySeqLen) {
                // Prefill 阶段：提取所有 head 的完整矩阵
                const allHeadWeights: number[][][] = [];
                for (let h = 0; h < numHeads; h++) {
                    allHeadWeights.push(extractHeadAttention(weights, h, querySeqLen, keySeqLen));
                }
                updatedHistory = {
                    lastKeySeqLen: keySeqLen,
                    numHeads,
                    weights: allHeadWeights,
                };
            } else {
                // Decode 阶段：querySeqLen = 1，需要累积到历史矩阵
                const prevHistory = history || { lastKeySeqLen: 0, numHeads, weights: [] };
                const allHeadWeights: number[][][] = [];

                for (let h = 0; h < numHeads; h++) {
                    // 获取当前 head 的新行
                    const headOffset = h * querySeqLen * keySeqLen;
                    const newRow: number[] = [];
                    for (let k = 0; k < keySeqLen; k++) {
                        newRow.push(weights[headOffset + k]);
                    }

                    // 获取之前的历史并扩展
                    const prevHeadWeights = prevHistory.weights[h] || [];
                    const updatedHeadWeights: number[][] = [];

                    for (const oldRow of prevHeadWeights) {
                        // 扩展旧行到新的 keySeqLen（新位置填0，因果掩码）
                        const extendedRow = [...oldRow];
                        while (extendedRow.length < keySeqLen) {
                            extendedRow.push(0);
                        }
                        updatedHeadWeights.push(extendedRow);
                    }
                    // 添加新行
                    updatedHeadWeights.push(newRow);

                    allHeadWeights.push(updatedHeadWeights);
                }

                updatedHistory = {
                    lastKeySeqLen: keySeqLen,
                    numHeads,
                    weights: allHeadWeights,
                };
            }

            attentionHistoryRef.current.set(selectedLayer, updatedHistory);

            // 根据 selectedHead 提取或计算平均
            const resultWeights = extractOrAverageFromHistory(updatedHistory, selectedHead);

            return {
                weights: resultWeights,
                numHeads,
                tokens: externalTokens || [],
                availableLayers: data.map((d) => d.layerIndex),
            };
        }

        // 旧格式: AttentionData
        const oldData = data as AttentionData;
        if (!oldData.weights) return null;

        const weights = oldData.weights;

        // 3D: [numHeads, seqLen, seqLen] -> 取选中的 head 或计算平均
        if (Array.isArray(weights[0]) && Array.isArray(weights[0][0])) {
            const weights3D = weights as number[][][];
            const numHeads = weights3D.length;

            let processedWeights: number[][];
            if (selectedHead === "average") {
                // 计算所有头的平均
                const seqLen = weights3D[0].length;
                processedWeights = Array.from({ length: seqLen }, () =>
                    Array(seqLen).fill(0)
                );
                for (let h = 0; h < numHeads; h++) {
                    for (let i = 0; i < seqLen; i++) {
                        for (let j = 0; j < weights3D[h][i].length; j++) {
                            processedWeights[i][j] += weights3D[h][i][j] / numHeads;
                        }
                    }
                }
            } else {
                const headIdx = Math.min(selectedHead, numHeads - 1);
                processedWeights = weights3D[headIdx];
            }

            return {
                weights: processedWeights,
                numHeads,
                tokens: oldData.tokens,
                availableLayers: undefined,
            };
        }

        // 2D: [seqLen, seqLen] -> 直接使用
        return {
            weights: weights as number[][],
            numHeads: 1,
            tokens: oldData.tokens,
            availableLayers: undefined,
        };
    }, [data, selectedLayer, selectedHead, externalTokens]);

    // 无数据状态
    if (!data || !processedData) {
        return (
            <div className="panel">
                <div className="panel-header">
                    <span className="text-cyber-purple">Attention Weights</span>
                </div>
                <div className="p-8 text-center text-gray-500">
                    <p>No Attention Data</p>
                    <p className="text-xs mt-2">
                        Attention weight visualization will appear after generation starts
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="panel">
            {/* Header with controls */}
            <div className="panel-header flex items-center justify-between">
                <span className="text-cyber-purple">
                    Attention Weights (Layer {selectedLayer + 1}/{numLayers})
                </span>
                <div className="flex items-center gap-3">
                    {/* Head selector with Average option */}
                    {processedData.numHeads > 1 && (
                        <select
                            value={selectedHead}
                            onChange={(e) => {
                                const val = e.target.value;
                                handleHeadChange(val === "average" ? "average" : Number(val));
                            }}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300"
                        >
                            <option value="average">Average</option>
                            {Array.from({ length: processedData.numHeads }, (_, i) => (
                                <option key={i} value={i}>
                                    Head {i + 1}
                                </option>
                            ))}
                        </select>
                    )}

                    {/* Layer selector */}
                    {onLayerChange && (
                        <select
                            value={selectedLayer}
                            onChange={(e) => onLayerChange(Number(e.target.value))}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300"
                        >
                            {processedData.availableLayers
                                ? processedData.availableLayers.map((layerIdx) => (
                                    <option key={layerIdx} value={layerIdx}>
                                        Layer {layerIdx + 1}
                                    </option>
                                ))
                                : Array.from({ length: numLayers }, (_, i) => (
                                    <option key={i} value={i}>
                                        Layer {i + 1}
                                    </option>
                                ))}
                        </select>
                    )}

                    {/* View mode toggle */}
                    <div className="flex rounded overflow-hidden border border-gray-700">
                        <button
                            onClick={() => setViewMode("links")}
                            className={`px-2 py-1 text-xs ${viewMode === "links"
                                ? "bg-cyber-purple text-white"
                                : "bg-gray-800 text-gray-400 hover:text-white"
                                }`}
                        >
                            Links
                        </button>
                        <button
                            onClick={() => setViewMode("heatmap")}
                            className={`px-2 py-1 text-xs ${viewMode === "heatmap"
                                ? "bg-cyber-purple text-white"
                                : "bg-gray-800 text-gray-400 hover:text-white"
                                }`}
                        >
                            Heatmap
                        </button>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="overflow-auto" style={{ maxHeight }}>
                {viewMode === "links" && (
                    <LinksView weights={processedData.weights} tokens={processedData.tokens} />
                )}
                {viewMode === "heatmap" && (
                    <HeatmapView
                        weights={processedData.weights}
                        tokens={processedData.tokens}
                        showValues={showValues}
                    />
                )}
            </div>
        </div>
    );
}
