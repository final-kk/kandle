import React from 'react';

export interface BaseStats {
  totalTime: number;
  tokenCount: number;
  tokensPerSecond: number;
  audioPreprocessTime?: number;
  encoderTime?: number;
  decoderTime?: number;
}

interface StatsDisplayProps {
  stats: BaseStats | null;
  modelType: 'qwen3' | 'whisper';
  className?: string;
}

/**
 * Statistics Display Component
 * Shows inference performance metrics
 */
export const StatsDisplay: React.FC<StatsDisplayProps> = ({
  stats,
  modelType,
  className = '',
}) => {
  if (!stats) {
    return null;
  }

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className={`panel ${className}`}>
      <div className="panel-header">Performance Metrics</div>
      <div className="p-3 grid grid-cols-3 gap-3">
        <div className="stat-card">
          <div className="stat-label">Total Time</div>
          <div className="stat-value">{formatTime(stats.totalTime)}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">
            {modelType === 'qwen3' ? 'Tokens' : 'Output Tokens'}
          </div>
          <div className="stat-value">{stats.tokenCount}</div>
        </div>

        <div className="stat-card">
          <div className="stat-label">Speed</div>
          <div className="stat-value">
            {stats.tokensPerSecond.toFixed(2)}
            <span className="text-xs text-gray-500 ml-1">tok/s</span>
          </div>
        </div>

        {/* Additional Whisper stats */}
        {modelType === 'whisper' && stats.audioPreprocessTime !== undefined && (
          <>
            <div className="stat-card">
              <div className="stat-label">Audio Preprocess</div>
              <div className="stat-value text-cyber-purple">
                {formatTime(stats.audioPreprocessTime)}
              </div>
            </div>
            {stats.encoderTime !== undefined && (
              <div className="stat-card">
                <div className="stat-label">Encoder</div>
                <div className="stat-value text-cyber-orange">
                  {formatTime(stats.encoderTime)}
                </div>
              </div>
            )}
            {stats.decoderTime !== undefined && (
              <div className="stat-card">
                <div className="stat-label">Decoder</div>
                <div className="stat-value text-cyber-pink">
                  {formatTime(stats.decoderTime)}
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {/* Visual speedometer */}
      <div className="px-3 pb-3">
        <div className="text-xs text-gray-500 mb-1">Speed Indicator</div>
        <div className="relative h-4 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-cyber-cyan via-cyber-purple to-cyber-pink"
            style={{
              width: `${Math.min(100, (stats.tokensPerSecond / 50) * 100)}%`,
              transition: 'width 0.5s ease-out',
            }}
          />
          {/* Markers */}
          <div className="absolute inset-0 flex justify-between px-2 text-[8px] text-gray-600">
            <span>0</span>
            <span>10</span>
            <span>20</span>
            <span>30</span>
            <span>40</span>
            <span>50</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatsDisplay;
