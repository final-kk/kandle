/**
 * LogitLensPanel - Logit Lens 可视化面板
 *
 * 显示各层 Transformer 的预测结果：
 * - 从下往上依次显示各层（模拟信息流动方向）
 * - 每层显示 Top-3 预测 token
 * - 动画效果：从下往上依次淡入
 */

import { useMemo } from "react";
import { LogitLensLayerCard } from "./LogitLensLayerCard";
import type { LayerPrediction } from "../../workers/message-types";

// ============================================================================
// Types
// ============================================================================

export interface LogitLensPanelProps {
    /** 各层的预测结果 */
    layerPredictions: LayerPrediction[] | null | undefined;

    /** 是否正在生成 */
    isGenerating: boolean;

    /** 是否正在加载 */
    isLoading?: boolean;

    /** 总层数 */
    totalLayers?: number;
}

// ============================================================================
// Component
// ============================================================================

export function LogitLensPanel({
    layerPredictions,
    isGenerating,
    isLoading = false,
    totalLayers = 28,
}: LogitLensPanelProps) {
    // 按层索引从大到小排序（底部是低层，顶部是高层）
    // 显示时从下往上，但数据顺序是从高层到低层
    const sortedPredictions = useMemo(() => {
        if (!layerPredictions || layerPredictions.length === 0) {
            return [];
        }
        // 按 layerIndex 降序排序，这样高层在上，低层在下
        return [...layerPredictions].sort((a, b) => b.layerIndex - a.layerIndex);
    }, [layerPredictions]);

    // 计算动画延迟（从下往上淡入）
    const getAnimationDelay = (index: number): number => {
        // index 0 是最高层（顶部），应该最后显示
        // 倒序计算延迟，底部的先显示
        const reversedIndex = sortedPredictions.length - 1 - index;
        return reversedIndex * 80; // 每层间隔 80ms
    };

    // 空状态
    if (!layerPredictions || layerPredictions.length === 0) {
        return (
            <div className="h-full flex items-center justify-center p-4">
                <div className="text-center text-gray-500 text-sm">
                    {isGenerating ? (
                        <div className="space-y-2">
                            <div className="w-5 h-5 border-2 border-cyber-purple border-t-transparent rounded-full animate-spin mx-auto" />
                            <p>Analyzing layer predictions...</p>
                        </div>
                    ) : (
                        <p>Layer predictions will appear after generation starts</p>
                    )}
                </div>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col">
            {/* Header info */}
            <div className="flex items-center justify-between px-2 py-1 flex-shrink-0">
                <span className="text-xs text-gray-500">({sortedPredictions.length} layers)</span>
                {isLoading && (
                    <div className="w-3 h-3 border border-cyber-purple border-t-transparent rounded-full animate-spin" />
                )}
            </div>

            {/* Layer Cards */}
            <div className="flex-1 overflow-y-auto p-2 space-y-2 min-h-0">
                {/* 说明：显示顺序 - 顶部是高层（接近输出），底部是低层（接近输入） */}
                <div className="text-[10px] text-gray-600 text-center mb-2">Output ↑</div>

                {sortedPredictions.map((prediction, index) => (
                    <LogitLensLayerCard
                        key={prediction.layerIndex}
                        layerIndex={prediction.layerIndex}
                        topK={prediction.topK}
                        animationDelay={getAnimationDelay(index)}
                        animate={true}
                        totalLayers={totalLayers}
                    />
                ))}

                <div className="text-[10px] text-gray-600 text-center mt-2">↓ Input</div>
            </div>

            {/* Footer info */}
            <div className="p-2 border-t border-gray-700/50 flex-shrink-0">
                <div className="text-[10px] text-gray-500 text-center">
                    Predictions after RMSNorm + LM Head projection at each layer
                </div>
            </div>
        </div>
    );
}
