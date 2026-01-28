/**
 * SamplerSettings - 采样参数设置
 *
 * 提供 Temperature、Top-K、Top-P 的滑块控制
 */

import type { SamplerConfig } from "../../types";

// ============================================================================
// Types
// ============================================================================

export interface SamplerSettingsProps {
    /** 当前配置 */
    config: SamplerConfig;

    /** 配置变更回调 */
    onChange: (config: Partial<SamplerConfig>) => void;

    /** 是否禁用（生成中） */
    disabled?: boolean;
}

// ============================================================================
// Component
// ============================================================================

export function SamplerSettings({ config, onChange, disabled = false }: SamplerSettingsProps) {
    return (
        <div className="panel">
            <div className="panel-header">
                <span className="text-cyber-cyan">Sampling Parameters</span>
            </div>
            <div className="p-3 space-y-4">
                {/* Temperature */}
                <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                        <label className="text-gray-300">Temperature</label>
                        <span className="font-mono text-cyber-cyan">
                            {config.temperature.toFixed(2)}
                        </span>
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="2"
                        step="0.05"
                        value={config.temperature}
                        onChange={(e) => onChange({ temperature: parseFloat(e.target.value) })}
                        disabled={disabled}
                        className="w-full accent-cyber-cyan"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                        <span>Deterministic</span>
                        <span>Creative</span>
                    </div>
                </div>

                {/* Top-K */}
                <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                        <label className="text-gray-300">Top-K</label>
                        <span className="font-mono text-cyber-cyan">
                            {config.topK === 0 ? "∞" : config.topK}
                        </span>
                    </div>
                    <input
                        type="range"
                        min="0"
                        max="100"
                        step="1"
                        value={config.topK}
                        onChange={(e) => onChange({ topK: parseInt(e.target.value) })}
                        disabled={disabled}
                        className="w-full accent-cyber-cyan"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                        <span>No limit</span>
                        <span>Top 100</span>
                    </div>
                </div>

                {/* Top-P */}
                <div className="space-y-1">
                    <div className="flex justify-between text-sm">
                        <label className="text-gray-300">Top-P (Nucleus)</label>
                        <span className="font-mono text-cyber-cyan">{config.topP.toFixed(2)}</span>
                    </div>
                    <input
                        type="range"
                        min="0.1"
                        max="1"
                        step="0.05"
                        value={config.topP}
                        onChange={(e) => onChange({ topP: parseFloat(e.target.value) })}
                        disabled={disabled}
                        className="w-full accent-cyber-cyan"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                        <span>Focused</span>
                        <span>Diverse</span>
                    </div>
                </div>

                {/* Greedy / Sample Toggle */}
                <div className="flex items-center gap-2 pt-2 border-t border-white/10">
                    <input
                        type="checkbox"
                        id="doSample"
                        checked={config.doSample}
                        onChange={(e) => onChange({ doSample: e.target.checked })}
                        disabled={disabled}
                        className="accent-cyber-cyan"
                    />
                    <label htmlFor="doSample" className="text-sm text-gray-300">
                        Enable Sampling
                    </label>
                    <span className="text-xs text-gray-500 ml-auto">
                        {config.doSample ? "Random" : "Greedy"}
                    </span>
                </div>
            </div>
        </div>
    );
}
