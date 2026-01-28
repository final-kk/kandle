/**
 * TokenProbsHUD - Token 概率分布可视化
 *
 * 核心可解释性组件，显示：
 * - Top-K 候选 token 及其概率
 * - 可点击选择干预生成
 * - 当前选中的 token 高亮
 */

import type { TokenCandidate } from "../../types";

// ============================================================================
// Types
// ============================================================================

export interface TokenProbsHUDProps {
    /** Top-K 候选 tokens */
    topK: TokenCandidate[];

    /** 当前选中的 token ID */
    selectedTokenId: number | null;

    /** 点击候选 token 时的回调 */
    onTokenSelect?: (tokenId: number) => void;

    /** 是否处于生成中状态 */
    isGenerating: boolean;

    /** 是否允许交互（干预选择） */
    interactive?: boolean;

    /** 是否正在加载下一步（禁用交互） */
    isLoading?: boolean;
}

// ============================================================================
// Component
// ============================================================================

export function TokenProbsHUD({
    topK,
    selectedTokenId,
    onTokenSelect,
    isGenerating,
    interactive = true,
    isLoading = false,
}: TokenProbsHUDProps) {
    // 加载时禁用交互
    const canInteract = interactive && !isLoading;

    if (topK.length === 0) {
        return (
            <div className="panel">
                <div className="panel-header">
                    <span className="text-cyber-cyan">Next-Token Prediction</span>
                </div>
                <div className="p-4 text-center text-gray-500">
                    {isGenerating ? "Computing..." : "Waiting for generation to start"}
                </div>
            </div>
        );
    }

    // 找出最大概率用于计算进度条宽度
    const maxProb = Math.max(...topK.map((t) => t.probability));

    return (
        <div className="panel relative">
            {/* Loading 遮罩 */}
            {isLoading && (
                <div className="absolute inset-0 bg-gray-900/70 flex items-center justify-center z-10 rounded-lg">
                    <div className="flex items-center gap-3">
                        <div className="w-5 h-5 border-2 border-cyber-cyan border-t-transparent rounded-full animate-spin" />
                        <span className="text-cyber-cyan text-sm">Computing next token...</span>
                    </div>
                </div>
            )}

            <div className="panel-header flex items-center justify-between">
                <span className="text-cyber-cyan">Next-Token Prediction</span>
                {canInteract && <span className="text-xs text-gray-400">Click to override</span>}
            </div>
            <div className="p-3 space-y-2">
                {topK.map((token, index) => {
                    const isSelected = token.tokenId === selectedTokenId;
                    const barWidth = (token.probability / maxProb) * 100;

                    return (
                        <div
                            key={token.tokenId}
                            className={`
                                relative group rounded transition-all duration-150
                                ${canInteract ? "cursor-pointer hover:bg-white/5" : "cursor-not-allowed opacity-70"}
                                ${isSelected ? "ring-1 ring-cyber-cyan" : ""}
                            `}
                            onClick={() => canInteract && onTokenSelect?.(token.tokenId)}
                        >
                            {/* 背景进度条 */}
                            <div
                                className={`
                                    absolute inset-y-0 left-0 rounded transition-all duration-300
                                    ${isSelected ? "bg-cyber-cyan/30" : "bg-cyber-purple/20"}
                                `}
                                style={{ width: `${barWidth}%` }}
                            />

                            {/* 内容 */}
                            <div className="relative flex items-center justify-between p-2">
                                <div className="flex items-center gap-2">
                                    {/* 排名 */}
                                    <span
                                        className={`
                                        text-xs w-5 text-center font-mono
                                        ${index === 0 ? "text-cyber-green" : "text-gray-500"}
                                    `}
                                    >
                                        {index + 1}
                                    </span>

                                    {/* Token 文本 */}
                                    <span
                                        className={`
                                        font-mono text-sm
                                        ${isSelected ? "text-cyber-cyan font-bold" : "text-white"}
                                    `}
                                    >
                                        {formatTokenText(token.text)}
                                    </span>

                                    {/* 选中标记 */}
                                    {isSelected && (
                                        <span className="text-cyber-green text-xs">← selected</span>
                                    )}
                                </div>

                                {/* 概率 */}
                                <span
                                    className={`
                                    font-mono text-sm
                                    ${isSelected ? "text-cyber-cyan" : "text-gray-400"}
                                `}
                                >
                                    {(token.probability * 100).toFixed(1)}%
                                </span>
                            </div>
                        </div>
                    );
                })}
            </div>

            {/* 底部提示 */}
            {canInteract && isGenerating && (
                <div className="px-3 pb-3">
                    <div className="text-xs text-gray-500 border-t border-white/10 pt-2">
                        Click any candidate token to override. The model will continue based on your selection.
                    </div>
                </div>
            )}
        </div>
    );
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * 格式化 token 文本显示
 * - 空格显示为可见字符
 * - 换行符显示为可见字符
 * - 特殊 token 高亮
 */
function formatTokenText(text: string): string {
    if (text.startsWith("<|") && text.endsWith("|>")) {
        // 特殊 token
        return text;
    }

    // 替换不可见字符
    return text
        .replace(/ /g, "␣") // 空格
        .replace(/\n/g, "↵") // 换行
        .replace(/\t/g, "→"); // Tab
}
