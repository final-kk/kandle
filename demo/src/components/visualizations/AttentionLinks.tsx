/**
 * AttentionLinks - Attention 权重可视化组件
 *
 * 显示当前 token 对之前所有 token 的 attention 权重分布。
 * 使用 SVG 绘制连线，线条粗细和颜色反映 attention 强度。
 *
 * 支持：
 * - 多层/多头选择
 * - 热力图模式
 * - 连线模式（显示当前 token 对历史 token 的关注）
 */

import { useState, useMemo, useRef, useEffect } from "react";

// ============================================================================
// Types
// ============================================================================

export interface AttentionData {
    /** attention 权重矩阵 [numHeads, seqLen, seqLen] 或 [seqLen, seqLen] */
    weights: number[][][] | number[][];
    /** 层索引 */
    layerIndex: number;
    /** token 文本列表 */
    tokens: string[];
}

export interface AttentionLinksProps {
    /** Attention 数据 */
    data: AttentionData | null;

    /** 可用的层数 */
    numLayers?: number;

    /** 当前选中的层 */
    selectedLayer?: number;

    /** 层切换回调 */
    onLayerChange?: (layer: number) => void;

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
// Heatmap Sub-component
// ============================================================================

interface HeatmapViewProps {
    weights: number[][];
    tokens: string[];
    showValues: boolean;
}

function HeatmapView({ weights, tokens, showValues }: HeatmapViewProps) {
    const seqLen = tokens.length;
    const cellSize = Math.min(40, Math.max(20, 400 / seqLen));

    return (
        <div className="overflow-auto">
            <div
                className="grid gap-px bg-gray-800"
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
                    <>
                        {/* Row label - key token */}
                        <div
                            key={`l-${qi}`}
                            className="bg-gray-900 p-1 text-xs text-gray-400 truncate flex items-center"
                            title={tokens[qi]}
                        >
                            {tokens[qi].slice(0, 8)}
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
                                title={`${tokens[qi]} -> ${tokens[ki]}: ${(weight * 100).toFixed(1)}%`}
                            >
                                {showValues && weight > 0.05 && (
                                    <span className="text-white/80 text-[10px]">
                                        {(weight * 100).toFixed(0)}
                                    </span>
                                )}
                            </div>
                        ))}
                    </>
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
    focusIndex?: number;
}

function LinksView({ weights, tokens, focusIndex }: LinksViewProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [tokenPositions, setTokenPositions] = useState<{ x: number; y: number }[]>([]);

    // 使用最后一个 token 作为焦点（当前生成的 token）
    const currentIndex = focusIndex ?? tokens.length - 1;
    const currentWeights = weights[currentIndex] || [];

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
                                ${
                                    isCurrent
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

export function AttentionLinks({
    data,
    numLayers = 28,
    selectedLayer = 0,
    onLayerChange,
    mode = "links",
    maxHeight = 400,
    showValues = true,
}: AttentionLinksProps) {
    const [viewMode, setViewMode] = useState<"heatmap" | "links">(mode);
    const [selectedHead, setSelectedHead] = useState(0);

    // 处理 attention 数据
    const processedWeights = useMemo(() => {
        if (!data?.weights) return null;

        const weights = data.weights;

        // 3D: [numHeads, seqLen, seqLen] -> 取选中的 head
        if (Array.isArray(weights[0]) && Array.isArray(weights[0][0])) {
            const numHeads = weights.length;
            const headIdx = Math.min(selectedHead, numHeads - 1);
            return {
                weights: weights[headIdx] as number[][],
                numHeads,
            };
        }

        // 2D: [seqLen, seqLen] -> 直接使用
        return {
            weights: weights as number[][],
            numHeads: 1,
        };
    }, [data, selectedHead]);

    // 无数据状态
    if (!data || !processedWeights) {
        return (
            <div className="panel">
                <div className="panel-header">
                    <span className="text-cyber-purple">Attention Weights</span>
                </div>
                <div className="p-8 text-center text-gray-500">
                    <p>No Attention Data</p>
                    <p className="text-xs mt-2">Attention weight visualization will appear after generation starts</p>
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
                    {/* Head selector */}
                    {processedWeights.numHeads > 1 && (
                        <select
                            value={selectedHead}
                            onChange={(e) => setSelectedHead(Number(e.target.value))}
                            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs text-gray-300"
                        >
                            {Array.from({ length: processedWeights.numHeads }, (_, i) => (
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
                            {Array.from({ length: numLayers }, (_, i) => (
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
                            className={`px-2 py-1 text-xs ${
                                viewMode === "links"
                                    ? "bg-cyber-purple text-white"
                                    : "bg-gray-800 text-gray-400 hover:text-white"
                            }`}
                        >
                            Links
                        </button>
                        <button
                            onClick={() => setViewMode("heatmap")}
                            className={`px-2 py-1 text-xs ${
                                viewMode === "heatmap"
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
                {viewMode === "heatmap" ? (
                    <HeatmapView
                        weights={processedWeights.weights}
                        tokens={data.tokens}
                        showValues={showValues}
                    />
                ) : (
                    <LinksView weights={processedWeights.weights} tokens={data.tokens} />
                )}
            </div>
        </div>
    );
}
