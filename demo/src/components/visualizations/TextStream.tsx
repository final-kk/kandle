/**
 * TextStream - 生成文本流可视化
 *
 * 显示：
 * - 输入 Prompt（灰色）
 * - 已生成的 Tokens（彩色，干预的高亮）
 * - 当前位置光标
 */

import type { GeneratedToken } from "../../types";

// ============================================================================
// Types
// ============================================================================

export interface TextStreamProps {
    /** 输入 prompt */
    prompt: string;

    /** 已生成的 token 列表 */
    generatedTokens: GeneratedToken[];

    /** 是否正在生成 */
    isGenerating: boolean;

    /** 是否显示详细信息（hover token 显示概率） */
    showDetails?: boolean;

    /** 点击 token 回调（用于查看该位置的 attention） */
    onTokenClick?: (index: number, token: GeneratedToken) => void;

    /** 最大高度 */
    maxHeight?: number;
}

// ============================================================================
// Component
// ============================================================================

export function TextStream({
    prompt,
    generatedTokens,
    isGenerating,
    showDetails = true,
    onTokenClick,
    maxHeight = 300,
}: TextStreamProps) {
    const hasContent = prompt.length > 0 || generatedTokens.length > 0;

    if (!hasContent) {
        return (
            <div className="panel">
                <div className="panel-header">
                    <span className="text-cyber-cyan">Generated Text</span>
                </div>
                <div className="p-4 text-center text-gray-500">Input prompt for start</div>
            </div>
        );
    }

    return (
        <div className="panel">
            <div className="panel-header flex items-center justify-between">
                <span className="text-cyber-cyan">Generated Text</span>
                <span className="text-xs text-gray-500 font-mono">
                    {generatedTokens.length} tokens
                </span>
            </div>
            <div
                className="p-4 pb-8 overflow-auto font-mono text-sm flex flex-wrap items-start gap-1"
                style={{ maxHeight }}
            >
                {/* Prompt 部分 */}
                {prompt && (
                    <span className="px-2 py-1 rounded bg-gray-800/50 text-gray-400 text-xs">
                        {prompt}
                    </span>
                )}

                {/* 生成的 tokens */}
                {generatedTokens.map((token, index) => (
                    <TokenSpan
                        key={index}
                        token={token}
                        index={index}
                        showDetails={showDetails}
                        onClick={onTokenClick ? () => onTokenClick(index, token) : undefined}
                    />
                ))}

                {/* 光标 */}
                {isGenerating && <span className="text-cyber-cyan animate-pulse">|</span>}
            </div>

            {/* 统计信息 */}
            {generatedTokens.length > 0 && (
                <div className="px-4 pb-3 border-t border-white/10">
                    <div className="pt-2 flex items-center gap-4 text-xs text-gray-500">
                        <span>
                            Tokens: <span className="text-white">{generatedTokens.length}</span>
                        </span>
                        <span>
                            Interventions:{" "}
                            <span
                                className={`${countInterventions(generatedTokens) > 0 ? "text-cyber-orange" : "text-white"}`}
                            >
                                {countInterventions(generatedTokens)}
                            </span>
                        </span>
                        <span>
                            Avg Prob:{" "}
                            <span className="text-cyber-green">
                                {(averageProbability(generatedTokens) * 100).toFixed(1)}%
                            </span>
                        </span>
                    </div>
                </div>
            )}
        </div>
    );
}

// ============================================================================
// Sub-components
// ============================================================================

interface TokenSpanProps {
    token: GeneratedToken;
    index: number;
    showDetails: boolean;
    onClick?: () => void;
}

function TokenSpan({ token, index, showDetails, onClick }: TokenSpanProps) {
    // 根据概率确定背景颜色强度
    const bgOpacity = Math.max(0.1, Math.min(0.4, token.probability * 0.5));
    const isHighProb = token.probability > 0.5;

    // 格式化显示文本
    const displayText = formatTokenForDisplay(token.text);

    return (
        <span
            className={`
                relative group inline-flex items-center
                px-1.5 py-0.5 mx-0.5 my-0.5 rounded text-xs font-mono
                transition-all cursor-default
                ${
                    token.isOverride
                        ? "bg-cyber-orange/30 text-cyber-orange ring-1 ring-cyber-orange/50"
                        : isHighProb
                          ? "bg-cyber-cyan/20 text-white"
                          : "bg-gray-700/50 text-gray-300"
                }
                ${onClick ? "cursor-pointer hover:ring-1 hover:ring-cyber-cyan" : ""}
            `}
            style={{
                backgroundColor: token.isOverride ? undefined : `rgba(0, 255, 255, ${bgOpacity})`,
            }}
            onClick={onClick}
            title={
                showDetails
                    ? `Token #${index + 1}: "${token.text}" (${(token.probability * 100).toFixed(1)}%)${token.isOverride ? " [Overridden]" : ""}`
                    : undefined
            }
        >
            {displayText}

            {/* 概率小标签 */}
            {showDetails && (
                <span
                    className={`
                    ml-1 text-[10px] opacity-60
                    ${token.isOverride ? "text-cyber-orange" : isHighProb ? "text-cyber-green" : "text-gray-400"}
                `}
                >
                    {(token.probability * 100).toFixed(0)}%
                </span>
            )}

            {/* 干预标记 */}
            {token.isOverride && <span className="ml-0.5 text-[10px] text-cyber-pink">*</span>}

            {/* Hover 详情 tooltip - 显示在下方避免被裁剪 */}
            {showDetails && (
                <span
                    className="
                    absolute top-full left-1/2 -translate-x-1/2 mt-2
                    px-2 py-1 rounded bg-gray-900 border border-gray-700
                    text-xs whitespace-nowrap shadow-lg
                    opacity-0 group-hover:opacity-100 transition-opacity
                    pointer-events-none z-50
                "
                >
                    <span className="text-gray-400">#{index + 1}</span>
                    <span className="mx-1 text-white">{formatTokenLabel(token.text)}</span>
                    <span
                        className={
                            token.probability > 0.5 ? "text-cyber-green" : "text-cyber-orange"
                        }
                    >
                        {(token.probability * 100).toFixed(1)}%
                    </span>
                    {token.isOverride && <span className="ml-1 text-cyber-pink">[I]</span>}
                </span>
            )}
        </span>
    );
}

// ============================================================================
// Helpers
// ============================================================================

/**
 * 格式化 token 用于显示
 * - 保留空格和换行的可见性
 */
function formatTokenForDisplay(text: string): string {
    // 特殊 token 直接显示
    if (text.startsWith("<|") && text.endsWith("|>")) {
        return ` ${text} `;
    }

    // 换行符保留
    return text;
}

/**
 * 格式化 token 用于标签（tooltip）
 */
function formatTokenLabel(text: string): string {
    return text.replace(/ /g, "[SPC]").replace(/\n/g, "[NL]").replace(/\t/g, "[TAB]");
}

/**
 * 计算干预次数
 */
function countInterventions(tokens: GeneratedToken[]): number {
    return tokens.filter((t) => t.isOverride).length;
}

/**
 * 计算平均概率
 */
function averageProbability(tokens: GeneratedToken[]): number {
    if (tokens.length === 0) return 0;
    const sum = tokens.reduce((acc, t) => acc + t.probability, 0);
    return sum / tokens.length;
}
