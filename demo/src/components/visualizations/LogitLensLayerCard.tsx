/**
 * LogitLensLayerCard - 单层 Logit Lens 预测卡片
 *
 * 显示单个 Transformer 层的预测结果：
 * - 层索引标识
 * - Top-K 预测 token 及概率
 * - 从下往上的淡入动画效果
 */

import { useEffect, useState } from "react";
import type { TokenCandidate } from "../../types";

// ============================================================================
// Types
// ============================================================================

export interface LogitLensLayerCardProps {
    /** 层索引 */
    layerIndex: number;

    /** Top-K 候选 tokens */
    topK: TokenCandidate[];

    /** 动画延迟（ms），用于实现从下往上的依次淡入 */
    animationDelay?: number;

    /** 是否显示动画 */
    animate?: boolean;

    /** 总层数，用于计算层位置描述 */
    totalLayers?: number;
}

// ============================================================================
// Component
// ============================================================================

export function LogitLensLayerCard({
    layerIndex,
    topK,
    animationDelay = 0,
    animate = true,
    totalLayers = 28,
}: LogitLensLayerCardProps) {
    const [isVisible, setIsVisible] = useState(!animate);

    // 动画触发
    useEffect(() => {
        if (!animate) {
            setIsVisible(true);
            return;
        }

        setIsVisible(false);
        const timer = setTimeout(() => {
            setIsVisible(true);
        }, animationDelay);

        return () => clearTimeout(timer);
    }, [animate, animationDelay, layerIndex]);

    // 计算层位置描述
    const getLayerLabel = (): string => {
        if (layerIndex === 0) return "Early";
        if (layerIndex === totalLayers - 1) return "Final";
        const ratio = layerIndex / (totalLayers - 1);
        if (ratio < 0.33) return "Early";
        if (ratio < 0.66) return "Middle";
        return "Late";
    };

    // 获取层颜色
    const getLayerColor = (): string => {
        const ratio = layerIndex / (totalLayers - 1);
        if (ratio < 0.33) return "text-cyber-purple";
        if (ratio < 0.66) return "text-cyber-orange";
        return "text-cyber-green";
    };

    if (topK.length === 0) {
        return null;
    }

    return (
        <div
            className={`
                transition-all duration-300 ease-out
                ${isVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}
            `}
        >
            <div className="bg-gray-900/50 rounded border border-gray-700/50 p-2">
                {/* 层标识 */}
                <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-1.5">
                        <span className={`text-xs font-mono font-bold ${getLayerColor()}`}>
                            L{layerIndex}
                        </span>
                        <span className="text-[10px] text-gray-500">{getLayerLabel()}</span>
                    </div>
                </div>

                {/* Top-K 预测 */}
                <div className="space-y-0.5">
                    {topK.slice(0, 3).map((token, idx) => (
                        <div
                            key={token.tokenId}
                            className="flex items-center justify-between text-xs"
                        >
                            <div className="flex items-center gap-1 min-w-0 flex-1">
                                <span
                                    className={`
                                        w-3 text-center font-mono
                                        ${idx === 0 ? "text-cyber-cyan" : "text-gray-600"}
                                    `}
                                >
                                    {idx + 1}
                                </span>
                                <span
                                    className={`
                                        font-mono truncate
                                        ${idx === 0 ? "text-white" : "text-gray-400"}
                                    `}
                                    title={token.text}
                                >
                                    {formatTokenText(token.text)}
                                </span>
                            </div>
                            <span
                                className={`
                                    font-mono ml-1 flex-shrink-0
                                    ${idx === 0 ? "text-cyber-cyan" : "text-gray-500"}
                                `}
                            >
                                {(token.probability * 100).toFixed(0)}%
                            </span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * 格式化 token 文本显示
 */
function formatTokenText(text: string): string {
    if (text.startsWith("<|") && text.endsWith("|>")) {
        return text;
    }

    return text
        .replace(/ /g, "\u2423") // 空格 → ␣
        .replace(/\n/g, "\u21B5") // 换行 → ↵
        .replace(/\t/g, "\u2192"); // Tab → →
}
