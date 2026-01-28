/**
 * LogitLensConfig - Logit Lens 配置面板
 *
 * 用于配置 Logit Lens 的层选择：
 * - 启用/禁用开关
 * - 预设方案（稀疏/全部/自定义）
 * - 层索引选择器
 */

import { useState, useMemo, useCallback } from "react";
import type { LogitLensConfig as LogitLensConfigType } from "../../workers/message-types";

// ============================================================================
// Types
// ============================================================================

export interface LogitLensConfigProps {
    /** 当前配置 */
    config: LogitLensConfigType;

    /** 配置变更回调 */
    onChange: (config: LogitLensConfigType) => void;

    /** 是否禁用（生成中时禁用） */
    disabled?: boolean;

    /** 模型总层数 */
    totalLayers?: number;
}

// 预设方案
type Preset = "sparse" | "all" | "custom";

// ============================================================================
// Presets
// ============================================================================

/**
 * 生成预设层索引
 */
function generatePresetLayers(preset: Preset, totalLayers: number): number[] {
    switch (preset) {
        case "sparse":
            // 稀疏采样：6层，均匀分布
            // 对于 28 层: [0, 5, 11, 16, 22, 27]
            const step = Math.floor((totalLayers - 1) / 5);
            return [0, step, step * 2, step * 3, step * 4, totalLayers - 1];
        case "all":
            // 全部层
            return Array.from({ length: totalLayers }, (_, i) => i);
        case "custom":
        default:
            // 默认返回稀疏采样
            return generatePresetLayers("sparse", totalLayers);
    }
}

/**
 * 检测当前层配置属于哪个预设
 */
function detectPreset(layers: number[], totalLayers: number): Preset {
    if (layers.length === totalLayers) {
        return "all";
    }
    const sparseLayers = generatePresetLayers("sparse", totalLayers);
    if (layers.length === sparseLayers.length && layers.every((l, i) => l === sparseLayers[i])) {
        return "sparse";
    }
    return "custom";
}

// ============================================================================
// Component
// ============================================================================

export function LogitLensConfig({
    config,
    onChange,
    disabled = false,
    totalLayers = 28,
}: LogitLensConfigProps) {
    // 检测当前预设
    const currentPreset = useMemo(
        () => detectPreset(config.layerIndices, totalLayers),
        [config.layerIndices, totalLayers]
    );

    // 本地状态：展开自定义选择器
    const [showCustomSelector, setShowCustomSelector] = useState(currentPreset === "custom");

    // 切换预设
    const handlePresetChange = useCallback(
        (preset: Preset) => {
            if (preset === "custom") {
                setShowCustomSelector(true);
                return;
            }
            setShowCustomSelector(false);
            onChange({
                ...config,
                layerIndices: generatePresetLayers(preset, totalLayers),
            });
        },
        [config, onChange, totalLayers]
    );

    // 切换单个层
    const handleLayerToggle = useCallback(
        (layerIndex: number) => {
            const currentSet = new Set(config.layerIndices);
            if (currentSet.has(layerIndex)) {
                currentSet.delete(layerIndex);
            } else {
                currentSet.add(layerIndex);
            }
            // 排序后更新
            const newLayers = Array.from(currentSet).sort((a, b) => a - b);
            onChange({
                ...config,
                layerIndices: newLayers,
            });
        },
        [config, onChange]
    );

    return (
        <div className="space-y-3">
            {/* 预设选择 */}
            <div className="space-y-1.5">
                <div className="text-xs text-gray-400">Layer Preset</div>
                <div className="flex gap-1">
                    <button
                        onClick={() => handlePresetChange("sparse")}
                        disabled={disabled}
                        className={`
                            flex-1 px-2 py-1 text-xs rounded transition-colors
                            ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                            ${currentPreset === "sparse" ? "bg-cyber-purple text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}
                        `}
                    >
                        Sparse (6)
                    </button>
                    <button
                        onClick={() => handlePresetChange("all")}
                        disabled={disabled}
                        className={`
                            flex-1 px-2 py-1 text-xs rounded transition-colors
                            ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                            ${currentPreset === "all" ? "bg-cyber-purple text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}
                        `}
                    >
                        All ({totalLayers})
                    </button>
                    <button
                        onClick={() => handlePresetChange("custom")}
                        disabled={disabled}
                        className={`
                            flex-1 px-2 py-1 text-xs rounded transition-colors
                            ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                            ${currentPreset === "custom" ? "bg-cyber-purple text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}
                        `}
                    >
                        Custom
                    </button>
                </div>
            </div>

            {/* 当前选择预览 */}
            <div className="text-xs text-gray-500">Layers: {config.layerIndices.join(", ")}</div>

            {/* 自定义层选择器 */}
            {showCustomSelector && (
                <div className="space-y-1.5">
                    <div className="text-xs text-gray-400">Select Layers</div>
                    <div className="grid grid-cols-7 gap-1">
                        {Array.from({ length: totalLayers }, (_, i) => {
                            const isSelected = config.layerIndices.includes(i);
                            return (
                                <button
                                    key={i}
                                    onClick={() => handleLayerToggle(i)}
                                    disabled={disabled}
                                    className={`
                                        w-full aspect-square text-[10px] font-mono rounded transition-colors
                                        ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                                        ${isSelected ? "bg-cyber-purple text-white" : "bg-gray-800 text-gray-500 hover:bg-gray-700"}
                                    `}
                                >
                                    {i}
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Top-K 设置 */}
            <div className="space-y-1.5">
                <div className="text-xs text-gray-400">Top-K per layer</div>
                <div className="flex gap-1">
                    {[1, 3, 5].map((k) => (
                        <button
                            key={k}
                            onClick={() => onChange({ ...config, topK: k })}
                            disabled={disabled}
                            className={`
                                flex-1 px-2 py-1 text-xs rounded transition-colors
                                ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                                ${config.topK === k ? "bg-cyber-purple text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}
                            `}
                        >
                            {k}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
}

// ============================================================================
// Default Config
// ============================================================================

export const DEFAULT_LOGIT_LENS_CONFIG: LogitLensConfigType = {
    enabled: true,
    layerIndices: generatePresetLayers("sparse", 28),
    topK: 3,
};
