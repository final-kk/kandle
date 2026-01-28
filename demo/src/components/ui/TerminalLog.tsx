import React, { useEffect, useRef } from 'react';

export interface LogEntry {
  id: number;
  timestamp: number;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error' | 'kernel' | 'debug';
}

interface TerminalLogProps {
  logs: LogEntry[];
  title?: string;
  className?: string;
  maxHeight?: number;
  showTimestamp?: boolean;
}

/**
 * Terminal-style Log Display Component
 * Shows real-time kernel execution logs with syntax highlighting
 */
export const TerminalLog: React.FC<TerminalLogProps> = ({
  logs,
  title = 'Console',
  className = '',
  maxHeight = 300,
  showTimestamp = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  const formatTimestamp = (ts: number) => {
    const date = new Date(ts);
    return `${date.getHours().toString().padStart(2, '0')}:${date
      .getMinutes()
      .toString()
      .padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}.${date
      .getMilliseconds()
      .toString()
      .padStart(3, '0')}`;
  };

  const getTypePrefix = (type: LogEntry['type']) => {
    switch (type) {
      case 'info':
        return { prefix: 'INFO', color: 'text-gray-400' };
      case 'success':
        return { prefix: ' OK ', color: 'text-cyber-green' };
      case 'warning':
        return { prefix: 'WARN', color: 'text-cyber-orange' };
      case 'error':
        return { prefix: 'ERR!', color: 'text-red-400' };
      case 'kernel':
        return { prefix: 'KERN', color: 'text-cyber-cyan' };
      case 'debug':
        return { prefix: 'DEBG', color: 'text-cyber-purple' };
    }
  };

  return (
    <div className={`panel ${className}`}>
      <div className="panel-header flex items-center gap-2">
        <span className="text-cyber-green">●</span>
        <span>{title}</span>
        <span className="ml-auto text-gray-600 text-xs">{logs.length} entries</span>
      </div>
      <div
        ref={containerRef}
        className="terminal-log p-3 overflow-auto bg-gray-950"
        style={{ maxHeight }}
      >
        {logs.length === 0 ? (
          <div className="text-gray-600 italic">Waiting for logs...</div>
        ) : (
          logs.map((log) => {
            const { prefix, color } = getTypePrefix(log.type);
            return (
              <div key={log.id} className="flex gap-2 hover:bg-gray-900/50">
                {showTimestamp && (
                  <span className="text-gray-700 select-none">
                    {formatTimestamp(log.timestamp)}
                  </span>
                )}
                <span className={`${color} font-bold select-none`}>[{prefix}]</span>
                <span className={`log-${log.type}`}>{log.message}</span>
              </div>
            );
          })
        )}
        <div className="text-cyber-cyan animate-pulse">▌</div>
      </div>
    </div>
  );
};

export default TerminalLog;
