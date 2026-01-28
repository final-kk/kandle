import React, { useRef, useEffect, useMemo } from 'react';

// Color map definitions
const colorMaps = {
  viridis: [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
    [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
    [180, 222, 44], [253, 231, 37],
  ],
  inferno: [
    [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
    [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
  ],
  plasma: [
    [13, 8, 135], [75, 3, 161], [126, 3, 168], [168, 34, 150],
    [203, 70, 121], [229, 107, 93], [248, 148, 65], [253, 195, 40],
  ],
  magma: [
    [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
    [181, 54, 122], [229, 80, 100], [251, 135, 97], [254, 194, 135],
  ],
  turbo: [
    [48, 18, 59], [67, 86, 171], [53, 147, 199], [36, 195, 164],
    [80, 233, 103], [163, 252, 67], [229, 231, 55], [252, 186, 55],
    [251, 115, 36], [219, 48, 30],
  ],
  cyber: [
    [0, 0, 20], [0, 50, 80], [0, 100, 120], [0, 150, 160],
    [0, 200, 200], [0, 245, 255], [100, 250, 255], [180, 255, 255],
  ],
} as const;

type ColorMapName = keyof typeof colorMaps;

interface HeatmapProps {
  data: Float32Array | number[];
  width: number;
  height: number;
  colorMap?: ColorMapName;
  title?: string;
  showColorbar?: boolean;
  className?: string;
}

function interpolateColor(
  colorMap: readonly (readonly number[])[],
  t: number
): [number, number, number] {
  t = Math.max(0, Math.min(1, t));
  const idx = t * (colorMap.length - 1);
  const i = Math.floor(idx);
  const f = idx - i;

  if (i >= colorMap.length - 1) {
    return colorMap[colorMap.length - 1] as [number, number, number];
  }

  const c0 = colorMap[i];
  const c1 = colorMap[i + 1];

  return [
    Math.round(c0[0] + f * (c1[0] - c0[0])),
    Math.round(c0[1] + f * (c1[1] - c0[1])),
    Math.round(c0[2] + f * (c1[2] - c0[2])),
  ];
}

/**
 * Heatmap visualization component using Canvas 2D
 * Optimized for displaying tensor data, attention weights, etc.
 */
export const Heatmap: React.FC<HeatmapProps> = ({
  data,
  width,
  height,
  colorMap = 'cyber',
  title,
  showColorbar = true,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const colorbarRef = useRef<HTMLCanvasElement>(null);

  // Normalize data
  const { normalizedData, min, max } = useMemo(() => {
    const arr = Array.isArray(data) ? data : Array.from(data);
    let min = Infinity;
    let max = -Infinity;

    for (const v of arr) {
      if (v < min) min = v;
      if (v > max) max = v;
    }

    const range = max - min || 1;
    const normalized = arr.map((v) => (v - min) / range);

    return { normalizedData: normalized, min, max };
  }, [data]);

  // Draw heatmap
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const colors = colorMaps[colorMap];
    const imgData = ctx.createImageData(width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        const t = normalizedData[idx] ?? 0;
        const [r, g, b] = interpolateColor(colors, t);

        const pixelIdx = idx * 4;
        imgData.data[pixelIdx] = r;
        imgData.data[pixelIdx + 1] = g;
        imgData.data[pixelIdx + 2] = b;
        imgData.data[pixelIdx + 3] = 255;
      }
    }

    ctx.putImageData(imgData, 0, 0);
  }, [normalizedData, width, height, colorMap]);

  // Draw colorbar
  useEffect(() => {
    if (!showColorbar) return;
    const canvas = colorbarRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const colors = colorMaps[colorMap];
    const barHeight = canvas.height;
    const barWidth = canvas.width;

    for (let y = 0; y < barHeight; y++) {
      const t = 1 - y / barHeight;
      const [r, g, b] = interpolateColor(colors, t);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(0, y, barWidth, 1);
    }
  }, [colorMap, showColorbar]);

  return (
    <div className={`vis-container ${className}`}>
      {title && (
        <div className="px-2 py-1 border-b border-gray-800 text-xs font-mono text-cyber-cyan uppercase">
          {title}
        </div>
      )}
      <div className="flex">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="flex-1"
          style={{ imageRendering: 'pixelated' }}
        />
        {showColorbar && (
          <div className="flex flex-col items-center ml-2 text-xs font-mono text-gray-500">
            <span>{max.toFixed(2)}</span>
            <canvas
              ref={colorbarRef}
              width={12}
              height={100}
              className="my-1 border border-gray-700"
            />
            <span>{min.toFixed(2)}</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default Heatmap;
