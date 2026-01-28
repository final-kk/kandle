import React from 'react';
import type { WebGPUInfo } from '../../services/backend';

interface SystemInfoProps {
  webgpuInfo: WebGPUInfo | null;
  modelLoaded: boolean;
  modelType: 'qwen3' | 'whisper' | null;
  className?: string;
}

/**
 * System Information Display Component
 * Shows WebGPU status, memory limits, and model status
 */
export const SystemInfo: React.FC<SystemInfoProps> = ({
  webgpuInfo,
  modelLoaded,
  modelType,
  className = '',
}) => {
  if (!webgpuInfo) {
    return (
      <div className={`panel ${className}`}>
        <div className="panel-header">System</div>
        <div className="p-4 text-center">
          <div className="text-gray-500">Initializing WebGPU...</div>
          <div className="mt-2 w-8 h-8 border-2 border-cyber-cyan border-t-transparent rounded-full animate-spin mx-auto"></div>
        </div>
      </div>
    );
  }

  if (!webgpuInfo.available) {
    return (
      <div className={`panel ${className}`}>
        <div className="panel-header">System</div>
        <div className="p-4">
          <div className="text-red-400 font-bold">⚠️ WebGPU Not Available</div>
          <p className="text-sm text-gray-500 mt-2">
            Please use a WebGPU-enabled browser (Chrome 113+, Edge 113+, Firefox Nightly)
          </p>
        </div>
      </div>
    );
  }

  const { supportsF16, limits } = webgpuInfo;

  return (
    <div className={`panel ${className}`}>
      <div className="panel-header flex items-center justify-between">
        <span>System</span>
        <span className="text-cyber-green text-xs">● Online</span>
      </div>
      <div className="p-3 space-y-3">
        {/* GPU Info */}
        <div className="stat-card">
          <div className="stat-label">WebGPU Status</div>
          <div className="stat-value text-sm text-cyber-green">
            ✓ Available
          </div>
          <div className="text-xs text-gray-600">
            F16 Support: {supportsF16 ? '✓ Enabled' : '✗ Disabled'}
          </div>
        </div>

        {/* Model Status */}
        <div className="stat-card">
          <div className="stat-label">Model Status</div>
          <div className={`stat-value ${modelLoaded ? 'text-cyber-green' : 'text-gray-600'}`}>
            {modelLoaded ? (
              <>
                <span className="inline-block w-2 h-2 bg-cyber-green rounded-full mr-2 animate-pulse"></span>
                {modelType === 'qwen3' ? 'Qwen3-0.6B' : 'Whisper-Tiny'}
              </>
            ) : (
              'Not Loaded'
            )}
          </div>
        </div>

        {/* Memory Limits */}
        <div className="stat-card">
          <div className="stat-label">Buffer Limits</div>
          <div className="space-y-1 text-xs font-mono">
            <div className="flex justify-between">
              <span className="text-gray-500">Max Buffer:</span>
              <span className="text-cyber-cyan">
                {limits.maxBufferSize
                  ? `${(limits.maxBufferSize / 1024 / 1024 / 1024).toFixed(1)} GB`
                  : 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Max Storage:</span>
              <span className="text-cyber-cyan">
                {limits.maxStorageBufferBindingSize
                  ? `${(limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB`
                  : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemInfo;
