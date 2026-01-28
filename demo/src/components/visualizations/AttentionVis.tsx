import React, { useRef, useEffect, useMemo, useState } from 'react';

interface AttentionVisProps {
  data: Float32Array | number[];
  shape: number[]; // [seq_len, seq_len] or [batch, heads, seq, seq]
  tokens?: string[];
  title?: string;
  className?: string;
  selectedHead?: number;
}

/**
 * Attention Weights Visualization Component
 * Displays attention patterns as an interactive heatmap
 */
export const AttentionVis: React.FC<AttentionVisProps> = ({
  data,
  shape,
  tokens = [],
  title = 'Attention Weights',
  className = '',
  selectedHead = 0,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number; value: number } | null>(null);

  // Process attention matrix
  const { matrix, seqLen } = useMemo(() => {
    const arr = Array.isArray(data) ? data : Array.from(data);

    let seqLen: number;
    let matrix: number[];

    if (shape.length === 2) {
      // [seq_len, seq_len]
      seqLen = shape[0];
      matrix = arr;
    } else if (shape.length === 4) {
      // [batch, heads, seq, seq] - extract specific head
      const [, , sl] = shape;
      seqLen = sl;
      const headSize = sl * sl;
      const headOffset = selectedHead * headSize;
      matrix = arr.slice(headOffset, headOffset + headSize);
    } else {
      // Assume square
      seqLen = Math.sqrt(arr.length);
      matrix = arr;
    }

    return { matrix, seqLen: Math.floor(seqLen) };
  }, [data, shape, selectedHead]);

  // Calculate canvas size based on sequence length
  const cellSize = Math.max(8, Math.min(20, 300 / seqLen));
  const canvasSize = seqLen * cellSize;
  const labelOffset = tokens.length > 0 ? 60 : 0;

  // Draw attention heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw cells
    for (let row = 0; row < seqLen; row++) {
      for (let col = 0; col < seqLen; col++) {
        const value = matrix[row * seqLen + col] ?? 0;

        // Color based on attention value (cyan to white gradient)
        const intensity = Math.min(1, Math.max(0, value));
        const r = Math.floor(intensity * 255);
        const g = Math.floor(180 + intensity * 75);
        const b = 255;
        const alpha = 0.3 + intensity * 0.7;

        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        ctx.fillRect(
          labelOffset + col * cellSize,
          labelOffset + row * cellSize,
          cellSize - 1,
          cellSize - 1
        );
      }
    }

    // Draw token labels if provided
    if (tokens.length > 0) {
      ctx.font = '10px monospace';
      ctx.fillStyle = '#00f5ff';

      // X-axis labels (top)
      for (let i = 0; i < Math.min(tokens.length, seqLen); i++) {
        ctx.save();
        ctx.translate(labelOffset + i * cellSize + cellSize / 2, labelOffset - 5);
        ctx.rotate(-Math.PI / 4);
        ctx.textAlign = 'left';
        ctx.fillText(tokens[i].slice(0, 6), 0, 0);
        ctx.restore();
      }

      // Y-axis labels (left)
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (let i = 0; i < Math.min(tokens.length, seqLen); i++) {
        ctx.fillText(
          tokens[i].slice(0, 8),
          labelOffset - 5,
          labelOffset + i * cellSize + cellSize / 2
        );
      }
    }

    // Draw grid lines
    ctx.strokeStyle = 'rgba(0, 245, 255, 0.1)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= seqLen; i++) {
      // Vertical
      ctx.beginPath();
      ctx.moveTo(labelOffset + i * cellSize, labelOffset);
      ctx.lineTo(labelOffset + i * cellSize, labelOffset + seqLen * cellSize);
      ctx.stroke();
      // Horizontal
      ctx.beginPath();
      ctx.moveTo(labelOffset, labelOffset + i * cellSize);
      ctx.lineTo(labelOffset + seqLen * cellSize, labelOffset + i * cellSize);
      ctx.stroke();
    }

    // Highlight diagonal (self-attention typically focuses here)
    ctx.strokeStyle = 'rgba(191, 0, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(labelOffset, labelOffset);
    ctx.lineTo(labelOffset + seqLen * cellSize, labelOffset + seqLen * cellSize);
    ctx.stroke();
  }, [matrix, seqLen, cellSize, tokens, labelOffset]);

  // Handle mouse hover
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX - labelOffset;
    const y = (e.clientY - rect.top) * scaleY - labelOffset;

    const col = Math.floor(x / cellSize);
    const row = Math.floor(y / cellSize);

    if (row >= 0 && row < seqLen && col >= 0 && col < seqLen) {
      const value = matrix[row * seqLen + col] ?? 0;
      setHoveredCell({ row, col, value });
    } else {
      setHoveredCell(null);
    }
  };

  return (
    <div className={`vis-container ${className}`}>
      <div className="px-2 py-1 border-b border-gray-800 flex justify-between items-center">
        <span className="text-xs font-mono text-cyber-cyan uppercase">{title}</span>
        <span className="text-xs text-gray-500">
          {seqLen}×{seqLen}
        </span>
      </div>
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={canvasSize + labelOffset}
          height={canvasSize + labelOffset}
          className="w-full cursor-crosshair"
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoveredCell(null)}
        />
        {hoveredCell && (
          <div
            className="absolute bg-gray-900/90 border border-cyber-cyan px-2 py-1 text-xs font-mono pointer-events-none"
            style={{
              left: '50%',
              bottom: '100%',
              transform: 'translateX(-50%)',
            }}
          >
            <div className="text-cyber-cyan">
              [{hoveredCell.row}, {hoveredCell.col}]
            </div>
            <div className="text-white">{hoveredCell.value.toFixed(4)}</div>
            {tokens.length > 0 && (
              <div className="text-gray-400">
                "{tokens[hoveredCell.row] || '?'}" → "{tokens[hoveredCell.col] || '?'}"
              </div>
            )}
          </div>
        )}
      </div>
      <div className="flex justify-between items-center px-2 py-1 text-xs font-mono border-t border-gray-800">
        <div className="flex items-center gap-2">
          <div className="w-4 h-2 bg-gradient-to-r from-gray-900 to-cyber-cyan"></div>
          <span className="text-gray-500">0 → 1</span>
        </div>
        <div className="flex items-center gap-1 text-cyber-purple">
          <span>━</span>
          <span className="text-gray-500">Diagonal</span>
        </div>
      </div>
    </div>
  );
};

export default AttentionVis;
