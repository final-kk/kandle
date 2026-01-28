import React, { useRef, useEffect } from 'react';

interface MelSpectrogramVisProps {
  data: Float32Array | number[];
  shape: number[]; // [batch, n_mels, n_frames]
  title?: string;
  className?: string;
}

/**
 * Mel Spectrogram Visualization Component
 * Specialized for audio feature visualization
 */
export const MelSpectrogramVis: React.FC<MelSpectrogramVisProps> = ({
  data,
  shape,
  title = 'Mel Spectrogram',
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const nMels = shape.length >= 2 ? shape[shape.length - 2] : 80;
  const nFrames = shape.length >= 1 ? shape[shape.length - 1] : Math.floor((data.length || 1) / nMels);

  // Scale for display
  const displayWidth = Math.min(600, nFrames);
  const displayHeight = Math.min(200, nMels * 2);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const arr = Array.isArray(data) ? data : Array.from(data);

    // Find min/max for normalization
    let min = Infinity;
    let max = -Infinity;
    for (const v of arr) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min || 1;

    // Create image
    const scaleX = displayWidth / nFrames;
    const scaleY = displayHeight / nMels;

    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, displayWidth, displayHeight);

    // Draw spectrogram
    for (let mel = 0; mel < nMels; mel++) {
      for (let frame = 0; frame < nFrames; frame++) {
        // Mel bins are usually displayed with low frequencies at bottom
        const dataIdx = (nMels - 1 - mel) * nFrames + frame;
        const value = arr[dataIdx] ?? 0;
        const normalized = (value - min) / range;

        // Plasma-like colormap
        const r = Math.floor(13 + normalized * 240);
        const g = Math.floor(8 + normalized * 140);
        const b = Math.floor(135 + normalized * (-75));

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(
          frame * scaleX,
          mel * scaleY,
          Math.ceil(scaleX),
          Math.ceil(scaleY)
        );
      }
    }

    // Draw time axis markers
    ctx.fillStyle = '#00f5ff';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';

    const timeStep = Math.floor(nFrames / 6);
    for (let i = 0; i <= 6; i++) {
      const frame = i * timeStep;
      const time = (frame * 0.01).toFixed(1); // Assuming ~100 fps
      ctx.fillText(`${time}s`, frame * scaleX, displayHeight - 5);
    }

    // Draw frequency axis label
    ctx.save();
    ctx.translate(10, displayHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Mel Bins', 0, 0);
    ctx.restore();
  }, [data, nMels, nFrames, displayWidth, displayHeight]);

  return (
    <div className={`vis-container ${className}`}>
      <div className="px-2 py-1 border-b border-gray-800 flex justify-between items-center">
        <span className="text-xs font-mono text-cyber-cyan uppercase">{title}</span>
        <span className="text-xs text-gray-500">
          {nMels} mels × {nFrames} frames
        </span>
      </div>
      <div className="p-2 overflow-x-auto">
        <canvas
          ref={canvasRef}
          width={displayWidth}
          height={displayHeight}
          className="border border-gray-800"
        />
      </div>
      <div className="flex justify-center gap-4 px-2 py-1 text-xs font-mono border-t border-gray-800">
        <span className="text-gray-500">Time →</span>
        <div className="flex items-center gap-2">
          <div className="w-16 h-2 bg-gradient-to-r from-purple-900 via-pink-500 to-yellow-300"></div>
          <span className="text-gray-500">Amplitude</span>
        </div>
      </div>
    </div>
  );
};

export default MelSpectrogramVis;
