import React, { useCallback, useState, useRef, useEffect } from "react";
import type { LoadProgress } from "../../workers/message-types";
import type { LoadMethod } from "../../services/model-loader";
import { formatBytes, formatSpeed } from "../../services/model-loader";
import type { ModelType } from "../../config";

interface ModelLoaderProps {
    modelType: ModelType;
    loadMethod: LoadMethod;
    onLoadMethodChange: (method: LoadMethod) => void;
    onLoad: (files: FileList | null, customUrls?: { tokenizer?: string; model?: string }) => void;
    isLoading: boolean;
    progress: LoadProgress | null;
    error: string | null;
    defaultUrls?: { tokenizer: string; model: string };
}

/**
 * Model Loader Component
 * Supports URL, WebFile API, and File Input loading methods
 * Requires both tokenizer.json and model.safetensors files
 */
export const ModelLoader: React.FC<ModelLoaderProps> = ({
    modelType: _modelType,
    loadMethod,
    onLoadMethodChange,
    onLoad,
    isLoading,
    progress,
    error,
    defaultUrls,
}) => {
    // modelType can be used for dynamic configurations in the future
    void _modelType;
    const [isDragOver, setIsDragOver] = useState(false);
    // Separate URLs for tokenizer and model - initialized with defaults
    const [tokenizerUrl, setTokenizerUrl] = useState(defaultUrls?.tokenizer || "");
    const [modelUrl, setModelUrl] = useState(defaultUrls?.model || "");
    const tokenizerInputRef = useRef<HTMLInputElement>(null);
    const modelInputRef = useRef<HTMLInputElement>(null);
    // Track selected files
    const [selectedTokenizer, setSelectedTokenizer] = useState<File | null>(null);
    const [selectedModel, setSelectedModel] = useState<File | null>(null);

    // Update URLs when defaultUrls change (e.g., when model size changes)
    useEffect(() => {
        if (defaultUrls) {
            setTokenizerUrl(defaultUrls.tokenizer);
            setModelUrl(defaultUrls.model);
        }
    }, [defaultUrls]);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(true);
    }, []);

    const handleDragLeave = useCallback(() => {
        setIsDragOver(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragOver(false);
        // Process dropped files
        const files = e.dataTransfer.files;
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const name = file.name.toLowerCase();
            if (name.includes("tokenizer") && name.endsWith(".json")) {
                setSelectedTokenizer(file);
            } else if (name.endsWith(".safetensors")) {
                setSelectedModel(file);
            }
        }
    }, []);

    const handleTokenizerFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setSelectedTokenizer(e.target.files[0]);
        }
    }, []);

    const handleModelFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files.length > 0) {
            setSelectedModel(e.target.files[0]);
        }
    }, []);

    // Load with selected files
    const handleLoadSelectedFiles = useCallback(() => {
        if (!selectedTokenizer || !selectedModel) {
            return;
        }
        // Create a synthetic FileList-like object
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(selectedTokenizer);
        dataTransfer.items.add(selectedModel);
        onLoad(dataTransfer.files);
    }, [selectedTokenizer, selectedModel, onLoad]);

    const handleUrlLoad = useCallback(() => {
        if (!tokenizerUrl || !modelUrl) {
            return;
        }
        onLoad(null, { tokenizer: tokenizerUrl, model: modelUrl });
    }, [onLoad, tokenizerUrl, modelUrl]);

    const handleWebFileAPI = useCallback(async () => {
        onLoad(null);
    }, [onLoad]);

    const canLoadFiles = selectedTokenizer && selectedModel;
    const canLoadUrls = tokenizerUrl && modelUrl;

    const progressPercent = progress
        ? progress.total > 0
            ? Math.round((progress.loaded / progress.total) * 100)
            : 0
        : 0;

    return (
        <div className="space-y-4">
            {/* Compatibility Notice */}
            <div className="text-xs text-yellow-500/80 bg-yellow-500/10 border border-yellow-500/30 rounded p-2">
                ⚠️ Currently only official repository models are supported. Quantized models and
                third-party converted models are not supported unless layer key / dtype are identical.
            </div>

            {/* Method Selector */}
            <div className="flex gap-2">
                {(["url", "webfile", "input"] as LoadMethod[]).map((method) => (
                    <button
                        key={method}
                        onClick={() => onLoadMethodChange(method)}
                        disabled={isLoading}
                        className={`flex-1 px-3 py-2 text-xs font-mono uppercase tracking-wider border transition-all
              ${
                  loadMethod === method
                      ? "border-cyber-cyan text-cyber-cyan bg-cyber-cyan/10"
                      : "border-gray-700 text-gray-500 hover:border-gray-600 hover:text-gray-400"
              }
              disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                        {method === "url" && "🌐 URL"}
                        {method === "webfile" && "📂 WebFile"}
                        {method === "input" && "📎 Upload"}
                    </button>
                ))}
            </div>

            {/* URL Input */}
            {loadMethod === "url" && (
                <div className="space-y-3">
                    <div className="space-y-1">
                        <label className="text-xs text-gray-500 font-mono">Tokenizer URL:</label>
                        <input
                            type="text"
                            placeholder="https://huggingface.co/.../tokenizer.json"
                            value={tokenizerUrl}
                            onChange={(e) => setTokenizerUrl(e.target.value)}
                            className="input-cyber"
                            disabled={isLoading}
                        />
                    </div>
                    <div className="space-y-1">
                        <label className="text-xs text-gray-500 font-mono">Model URL:</label>
                        <input
                            type="text"
                            placeholder="https://huggingface.co/.../model.safetensors"
                            value={modelUrl}
                            onChange={(e) => setModelUrl(e.target.value)}
                            className="input-cyber"
                            disabled={isLoading}
                        />
                    </div>
                    <button
                        onClick={handleUrlLoad}
                        disabled={isLoading || !canLoadUrls}
                        className="btn-cyber w-full"
                    >
                        {isLoading
                            ? "Loading..."
                            : !canLoadUrls
                              ? "Please fill in both URLs"
                              : "Load from URL"}
                    </button>
                </div>
            )}

            {/* WebFile API */}
            {loadMethod === "webfile" && (
                <div className="space-y-2">
                    <p className="text-xs text-gray-500">
                        Use File System Access API to select local files (select both tokenizer.json
                        and model.safetensors)
                    </p>
                    <button
                        onClick={handleWebFileAPI}
                        disabled={isLoading}
                        className="btn-cyber w-full"
                    >
                        {isLoading ? "Loading..." : "Choose Files (hold Ctrl to multi-select)"}
                    </button>
                </div>
            )}

            {/* File Input - Separate file selectors */}
            {loadMethod === "input" && (
                <div className="space-y-3">
                    {/* Tokenizer File */}
                    <div className="space-y-1">
                        <label className="text-xs text-gray-500 font-mono">Tokenizer File:</label>
                        <div className="flex gap-2">
                            <button
                                onClick={() => tokenizerInputRef.current?.click()}
                                disabled={isLoading}
                                className="btn-cyber flex-1 text-left"
                            >
                                {selectedTokenizer
                                    ? `✓ ${selectedTokenizer.name}`
                                    : "Select tokenizer.json"}
                            </button>
                            <input
                                ref={tokenizerInputRef}
                                type="file"
                                accept=".json"
                                onChange={handleTokenizerFileSelect}
                                className="hidden"
                                disabled={isLoading}
                            />
                        </div>
                    </div>

                    {/* Model File */}
                    <div className="space-y-1">
                        <label className="text-xs text-gray-500 font-mono">Model File:</label>
                        <div className="flex gap-2">
                            <button
                                onClick={() => modelInputRef.current?.click()}
                                disabled={isLoading}
                                className="btn-cyber flex-1 text-left"
                            >
                                {selectedModel
                                    ? `✓ ${selectedModel.name}`
                                    : "Select model.safetensors"}
                            </button>
                            <input
                                ref={modelInputRef}
                                type="file"
                                accept=".safetensors"
                                onChange={handleModelFileSelect}
                                className="hidden"
                                disabled={isLoading}
                            />
                        </div>
                    </div>

                    {/* Drag & Drop Zone */}
                    <div
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                        className={`drop-zone ${isDragOver ? "drag-over" : ""}`}
                    >
                        <div className="text-2xl">📦</div>
                        <div className="text-xs text-gray-400">Or drag files here</div>
                    </div>

                    {/* Load Button */}
                    <button
                        onClick={handleLoadSelectedFiles}
                        disabled={isLoading || !canLoadFiles}
                        className="btn-cyber w-full"
                    >
                        {isLoading ? "Loading..." : !canLoadFiles ? "Please select both files" : "Load Model"}
                    </button>
                </div>
            )}

            {/* Progress Bar */}
            {isLoading && progress && (
                <div className="space-y-2 animate-fade-in">
                    <div className="flex justify-between text-xs font-mono">
                        <span className="text-cyber-cyan">
                            {progress.stage === "tokenizer"
                                ? "📝 Tokenizer"
                                : progress.stage === "weights"
                                  ? "⚙️ Weights"
                                  : "🧠 Model"}
                        </span>
                        <span className="text-gray-500">{progress.fileName}</span>
                    </div>
                    <div className="progress-cyber">
                        <div
                            className="progress-cyber-bar"
                            style={{ width: `${progressPercent}%` }}
                        />
                    </div>
                    <div className="flex justify-between text-xs font-mono text-gray-500">
                        <span>
                            {formatBytes(progress.loaded)} / {formatBytes(progress.total)}
                        </span>
                        <span>{progressPercent}%</span>
                        <span>{formatSpeed(progress.speed ?? 0)}</span>
                    </div>

                    {/* Terminal-style progress */}
                    <div className="bg-gray-900 p-2 rounded font-mono text-xs text-cyber-green">
                        <span className="text-gray-600">$</span> downloading {progress.fileName}...
                        <br />
                        <span className="text-gray-600">$</span>{" "}
                        {"█".repeat(Math.floor(progressPercent / 5))}
                        {"░".repeat(20 - Math.floor(progressPercent / 5))} {progressPercent}%
                    </div>
                </div>
            )}

            {/* Error Display */}
            {error && (
                <div className="p-3 bg-red-900/20 border border-red-900 rounded text-sm text-red-400 animate-fade-in">
                    <span className="font-bold">Error:</span> {error}
                </div>
            )}
        </div>
    );
};

export default ModelLoader;
