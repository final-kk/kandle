import React, { useRef, useEffect, useMemo } from 'react';

interface FeatureMapProps {
  data: Float32Array | number[];
  shape: number[]; // [batch, features] or [batch, seq, features]
  title?: string;
  method?: 'pca' | 'tsne' | 'umap' | 'raw';
  className?: string;
  width?: number;
  height?: number;
}

/**
 * Simple PCA implementation for 2D projection
 */
function simplePCA(data: number[], numFeatures: number, numSamples: number): { x: number[]; y: number[] } {
  // Center the data
  const means = new Array(numFeatures).fill(0);
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      means[j] += data[i * numFeatures + j];
    }
  }
  for (let j = 0; j < numFeatures; j++) {
    means[j] /= numSamples;
  }

  const centered = new Array(numSamples * numFeatures);
  for (let i = 0; i < numSamples; i++) {
    for (let j = 0; j < numFeatures; j++) {
      centered[i * numFeatures + j] = data[i * numFeatures + j] - means[j];
    }
  }

  // Use power iteration to find first two principal components
  // This is a simplified version for visualization
  const pc1 = new Array(numFeatures).fill(0).map(() => Math.random() - 0.5);
  const pc2 = new Array(numFeatures).fill(0).map(() => Math.random() - 0.5);

  // Normalize
  const normalize = (v: number[]) => {
    const norm = Math.sqrt(v.reduce((a, b) => a + b * b, 0));
    return v.map((x) => x / (norm || 1));
  };

  // Power iteration for PC1
  for (let iter = 0; iter < 20; iter++) {
    const newPc = new Array(numFeatures).fill(0);
    for (let i = 0; i < numSamples; i++) {
      let dot = 0;
      for (let j = 0; j < numFeatures; j++) {
        dot += centered[i * numFeatures + j] * pc1[j];
      }
      for (let j = 0; j < numFeatures; j++) {
        newPc[j] += centered[i * numFeatures + j] * dot;
      }
    }
    const normalized = normalize(newPc);
    for (let j = 0; j < numFeatures; j++) {
      pc1[j] = normalized[j];
    }
  }

  // Orthogonalize PC2
  let dot = 0;
  for (let j = 0; j < numFeatures; j++) {
    dot += pc2[j] * pc1[j];
  }
  for (let j = 0; j < numFeatures; j++) {
    pc2[j] -= dot * pc1[j];
  }
  const normalizedPc2 = normalize(pc2);
  for (let j = 0; j < numFeatures; j++) {
    pc2[j] = normalizedPc2[j];
  }

  // Power iteration for PC2
  for (let iter = 0; iter < 20; iter++) {
    const newPc = new Array(numFeatures).fill(0);
    for (let i = 0; i < numSamples; i++) {
      let dotVal = 0;
      for (let j = 0; j < numFeatures; j++) {
        dotVal += centered[i * numFeatures + j] * pc2[j];
      }
      for (let j = 0; j < numFeatures; j++) {
        newPc[j] += centered[i * numFeatures + j] * dotVal;
      }
    }
    // Orthogonalize
    let d = 0;
    for (let j = 0; j < numFeatures; j++) {
      d += newPc[j] * pc1[j];
    }
    for (let j = 0; j < numFeatures; j++) {
      newPc[j] -= d * pc1[j];
    }
    const normalized = normalize(newPc);
    for (let j = 0; j < numFeatures; j++) {
      pc2[j] = normalized[j];
    }
  }

  // Project data
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i < numSamples; i++) {
    let px = 0,
      py = 0;
    for (let j = 0; j < numFeatures; j++) {
      px += centered[i * numFeatures + j] * pc1[j];
      py += centered[i * numFeatures + j] * pc2[j];
    }
    x.push(px);
    y.push(py);
  }

  return { x, y };
}

/**
 * Feature Map visualization component
 * Projects high-dimensional features to 2D for visualization
 */
export const FeatureMap: React.FC<FeatureMapProps> = ({
  data,
  shape,
  title,
  method = 'pca',
  className = '',
  width = 300,
  height = 300,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Process data
  const projectedPoints = useMemo(() => {
    const arr = Array.isArray(data) ? data : Array.from(data);

    // Determine dimensions
    let numSamples: number;
    let numFeatures: number;

    if (shape.length === 2) {
      [numSamples, numFeatures] = shape;
    } else if (shape.length === 3) {
      // Flatten batch and seq
      numSamples = shape[0] * shape[1];
      numFeatures = shape[2];
    } else {
      numSamples = arr.length;
      numFeatures = 1;
    }

    if (numSamples === 0 || numFeatures === 0) {
      return { x: [], y: [], labels: [] };
    }

    if (method === 'raw' && numFeatures >= 2) {
      // Just use first two features
      const x: number[] = [];
      const y: number[] = [];
      for (let i = 0; i < numSamples; i++) {
        x.push(arr[i * numFeatures]);
        y.push(arr[i * numFeatures + 1]);
      }
      return { x, y, labels: Array.from({ length: numSamples }, (_, i) => i) };
    }

    // Use PCA
    const { x, y } = simplePCA(arr, numFeatures, numSamples);
    return { x, y, labels: Array.from({ length: numSamples }, (_, i) => i) };
  }, [data, shape, method]);

  // Draw scatter plot
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { x, y } = projectedPoints;
    if (x.length === 0) return;

    // Clear
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, width, height);

    // Draw grid
    ctx.strokeStyle = 'rgba(0, 245, 255, 0.1)';
    ctx.lineWidth = 1;
    const gridSize = 40;
    for (let i = 0; i <= width; i += gridSize) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, height);
      ctx.stroke();
    }
    for (let i = 0; i <= height; i += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(width, i);
      ctx.stroke();
    }

    // Normalize to canvas coordinates
    const padding = 30;
    const minX = Math.min(...x);
    const maxX = Math.max(...x);
    const minY = Math.min(...y);
    const maxY = Math.max(...y);
    const rangeX = maxX - minX || 1;
    const rangeY = maxY - minY || 1;

    const toCanvasX = (v: number) =>
      padding + ((v - minX) / rangeX) * (width - 2 * padding);
    const toCanvasY = (v: number) =>
      height - padding - ((v - minY) / rangeY) * (height - 2 * padding);

    // Draw points with gradient colors
    const numPoints = x.length;
    for (let i = 0; i < numPoints; i++) {
      const t = i / Math.max(1, numPoints - 1);
      const hue = 180 + t * 60; // Cyan to purple gradient

      ctx.beginPath();
      ctx.arc(toCanvasX(x[i]), toCanvasY(y[i]), 3, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${hue}, 100%, 60%, 0.8)`;
      ctx.fill();

      // Draw connecting lines for sequential data
      if (i > 0) {
        ctx.beginPath();
        ctx.moveTo(toCanvasX(x[i - 1]), toCanvasY(y[i - 1]));
        ctx.lineTo(toCanvasX(x[i]), toCanvasY(y[i]));
        ctx.strokeStyle = `hsla(${hue}, 100%, 50%, 0.2)`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // Draw start and end markers
    if (numPoints > 0) {
      // Start marker
      ctx.beginPath();
      ctx.arc(toCanvasX(x[0]), toCanvasY(y[0]), 6, 0, Math.PI * 2);
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 2;
      ctx.stroke();

      // End marker
      ctx.beginPath();
      ctx.arc(toCanvasX(x[numPoints - 1]), toCanvasY(y[numPoints - 1]), 6, 0, Math.PI * 2);
      ctx.strokeStyle = '#ff00aa';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [projectedPoints, width, height]);

  return (
    <div className={`vis-container ${className}`}>
      {title && (
        <div className="px-2 py-1 border-b border-gray-800 text-xs font-mono text-cyber-cyan uppercase">
          {title}
        </div>
      )}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="w-full"
      />
      <div className="flex justify-between px-2 py-1 text-xs text-gray-500 font-mono">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-cyber-green"></span>
          Start
        </span>
        <span>{projectedPoints.x.length} points</span>
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-cyber-pink"></span>
          End
        </span>
      </div>
    </div>
  );
};

export default FeatureMap;
