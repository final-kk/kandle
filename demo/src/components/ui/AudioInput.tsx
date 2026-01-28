import React, { useCallback, useRef, useState } from 'react';

interface AudioInputProps {
  onAudioLoad: (audioBuffer: ArrayBuffer) => void;
  disabled?: boolean;
  className?: string;
}

/**
 * Audio Input Component
 * For audio file upload (P0) with future microphone support (P1)
 */
export const AudioInput: React.FC<AudioInputProps> = ({
  onAudioLoad,
  disabled = false,
  className = '',
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setFileName(file.name);
      const buffer = await file.arrayBuffer();
      onAudioLoad(buffer);

      // Try to get duration using Web Audio API
      try {
        const audioContext = new AudioContext();
        const audioBuffer = await audioContext.decodeAudioData(buffer.slice(0));
        setDuration(audioBuffer.duration);
        audioContext.close();
      } catch {
        setDuration(null);
      }
    },
    [onAudioLoad]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith('audio/')) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`drop-zone ${isDragOver ? 'drag-over' : ''} ${
          disabled ? 'opacity-50 cursor-not-allowed' : ''
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.wav,.mp3,.m4a,.webm"
          onChange={handleFileSelect}
          className="hidden"
          disabled={disabled}
        />
        <div className="text-4xl">🎤</div>
        <div className="text-sm text-gray-400">
          Drag and drop audio file here, or click to select
        </div>
        <div className="text-xs text-gray-600">
          Supports WAV, MP3, M4A (16kHz WAV recommended)
        </div>
      </div>

      {/* Selected File Info */}
      {fileName && (
        <div className="flex items-center gap-3 p-3 bg-gray-800/50 rounded border border-gray-700">
          <div className="text-2xl">🎵</div>
          <div className="flex-1 min-w-0">
            <div className="font-mono text-sm text-cyber-cyan truncate">
              {fileName}
            </div>
            {duration !== null && (
              <div className="text-xs text-gray-500">
                Duration: {duration.toFixed(2)}s
              </div>
            )}
          </div>
          <button
            onClick={() => {
              setFileName(null);
              setDuration(null);
            }}
            className="text-gray-500 hover:text-gray-300"
          >
            ✕
          </button>
        </div>
      )}

      {/* Microphone Button (P1 - Coming Soon) */}
      <div className="flex items-center gap-2 text-gray-600 text-sm">
        <button
          disabled
          className="btn-cyber opacity-50 cursor-not-allowed flex items-center gap-2"
          title="Coming Soon"
        >
          <span>🎙️</span>
          <span>Record</span>
        </button>
        <span className="text-xs">(Coming Soon)</span>
      </div>
    </div>
  );
};

export default AudioInput;
