/**
 * GenerationControls - 生成控制面板
 *
 * 提供单步/自动/暂停/停止等生成控制
 */

import type { GeneratorStatus } from "../../types";

// ============================================================================
// Types
// ============================================================================

export interface GenerationControlsProps {
    /** 当前生成状态 */
    status: GeneratorStatus;

    /** 单步生成 */
    onStep: () => void;

    /** 后退一步 */
    onUndo: () => void;

    /** 是否可以后退 */
    canUndo: boolean;

    /** 自动生成 */
    onAutoPlay: () => void;

    /** 暂停 */
    onPause: () => void;

    /** 停止 */
    onStop: () => void;

    /** 重置 */
    onReset: () => void;

    /** 是否禁用（例如模型未加载） */
    disabled?: boolean;
}

// ============================================================================
// Component
// ============================================================================

export function GenerationControls({
    status,
    onStep,
    onUndo,
    canUndo,
    onAutoPlay,
    onPause,
    onStop,
    onReset,
    disabled = false,
}: GenerationControlsProps) {
    const isIdle = status === "idle";
    const isGenerating = status === "generating";
    const isPaused = status === "paused";
    const isComplete = status === "complete";
    const isError = status === "error";

    const canStep = (isGenerating || isPaused) && !disabled;
    const canAutoPlay = isGenerating && !disabled;
    const canPause = isGenerating && !disabled;
    const canStop = (isGenerating || isPaused) && !disabled;
    const canReset = (isComplete || isError || isPaused) && !disabled;

    return (
        <div className="panel">
            <div className="panel-header">
                <span className="text-cyber-cyan">Generation Control</span>
                <StatusBadge status={status} />
            </div>
            <div className="p-3">
                <div className="flex flex-wrap gap-2">
                    {/* Step */}
                    <button
                        className={`btn-cyber ${canStep ? "btn-cyber-primary" : ""}`}
                        onClick={onStep}
                        disabled={!canStep}
                        title="Generate next token"
                    >
                        <span className="mr-1">▶</span>
                        Step
                    </button>

                    {/* Undo */}
                    <button
                        className={`btn-cyber ${canUndo ? "btn-cyber-secondary" : ""}`}
                        onClick={onUndo}
                        disabled={!canUndo || disabled}
                        title="Undo last token and regenerate"
                    >
                        <span className="mr-1">↶</span>
                        Undo
                    </button>

                    {/* Auto Play */}
                    <button
                        className={`btn-cyber ${canAutoPlay ? "" : ""}`}
                        onClick={onAutoPlay}
                        disabled={!canAutoPlay}
                        title="Auto-generate continuously"
                    >
                        <span className="mr-1">⏭</span>
                        Auto
                    </button>

                    {/* Pause */}
                    <button
                        className="btn-cyber"
                        onClick={onPause}
                        disabled={!canPause}
                        title="Pause auto-generation"
                    >
                        <span className="mr-1">⏸</span>
                        Pause
                    </button>

                    {/* Stop */}
                    <button
                        className="btn-cyber"
                        onClick={onStop}
                        disabled={!canStop}
                        title="Stop generation"
                    >
                        <span className="mr-1">■</span>
                        Stop
                    </button>

                    {/* Reset */}
                    <button
                        className="btn-cyber"
                        onClick={onReset}
                        disabled={!canReset}
                        title="Reset and start new generation"
                    >
                        <span className="mr-1">↺</span>
                        Reset
                    </button>
                </div>

                {/* Status Hint */}
                <div className="mt-3 text-xs text-gray-500">
                    {isIdle && "Enter prompt and click Start to begin generation"}
                    {isGenerating && "Click Step to generate next token, or click candidate to override"}
                    {isPaused && "Generation paused, click Step to continue"}
                    {isComplete && "Generation complete, click Reset to start new conversation"}
                    {isError && "Generation error, please check model status"}
                </div>
            </div>
        </div>
    );
}

// ============================================================================
// Sub-components
// ============================================================================

function StatusBadge({ status }: { status: GeneratorStatus }) {
    const config = {
        idle: { text: "IDLE", color: "text-gray-400" },
        generating: { text: "RUNNING", color: "text-cyber-green" },
        paused: { text: "PAUSED", color: "text-cyber-orange" },
        complete: { text: "DONE", color: "text-cyber-cyan" },
        error: { text: "ERROR", color: "text-cyber-pink" },
    }[status];

    return <span className={`text-xs font-mono ${config.color}`}>[{config.text}]</span>;
}
