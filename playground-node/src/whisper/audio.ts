/**
 * Whisper Audio Preprocessing
 *
 * WAV 文件解析和 Mel 频谱图计算
 *
 * 工作流程：
 * 1. 解析 WAV 文件 (RIFF 头)
 * 2. 提取 PCM 数据
 * 3. 重采样到 16kHz
 * 4. 计算 Mel 频谱图
 * 5. 应用 Whisper 特定的 log 归一化
 *
 * @module playground-node/whisper/audio
 */

import * as fs from "fs/promises";
import { Tensor, audio } from "@kandle/core";
import { WHISPER_AUDIO_CONFIG } from "@kandle/model-utils";

// ============================================================================
// Types
// ============================================================================

/**
 * WAV 文件头信息
 */
export interface WavHeader {
    /** 音频格式 (1 = PCM) */
    audioFormat: number;
    /** 声道数 */
    numChannels: number;
    /** 采样率 */
    sampleRate: number;
    /** 字节率 */
    byteRate: number;
    /** 块对齐 */
    blockAlign: number;
    /** 位深度 */
    bitsPerSample: number;
    /** 数据大小 */
    dataSize: number;
    /** 数据起始偏移 */
    dataOffset: number;
}

/**
 * 解析后的音频数据
 */
export interface ParsedAudio {
    /** WAV 头信息 */
    header: WavHeader;
    /** PCM 采样数据 (Float32, 单声道, 范围 [-1, 1]) */
    samples: Float32Array;
}

// ============================================================================
// WAV 解析
// ============================================================================

/**
 * 解析 WAV 文件头
 *
 * WAV 文件结构:
 * - RIFF 头 (12 bytes)
 * - fmt  子块 (至少 16 bytes)
 * - data 子块
 *
 * @param buffer - WAV 文件的 ArrayBuffer
 * @returns WAV 头信息
 */
export function parseWavHeader(buffer: ArrayBuffer): WavHeader {
    const view = new DataView(buffer);

    // ==========================================
    // RIFF 头 (12 bytes)
    // ==========================================
    // 0-3: "RIFF"
    const riff = String.fromCharCode(
        view.getUint8(0),
        view.getUint8(1),
        view.getUint8(2),
        view.getUint8(3)
    );
    if (riff !== "RIFF") {
        throw new Error(`Invalid WAV file: expected RIFF, got ${riff}`);
    }

    // 4-7: 文件大小 (不包括 RIFF 头的 8 字节)
    // const fileSize = view.getUint32(4, true);

    // 8-11: "WAVE"
    const wave = String.fromCharCode(
        view.getUint8(8),
        view.getUint8(9),
        view.getUint8(10),
        view.getUint8(11)
    );
    if (wave !== "WAVE") {
        throw new Error(`Invalid WAV file: expected WAVE, got ${wave}`);
    }

    // ==========================================
    // 查找 fmt 和 data 子块
    // ==========================================
    let offset = 12;
    let audioFormat = 0;
    let numChannels = 0;
    let sampleRate = 0;
    let byteRate = 0;
    let blockAlign = 0;
    let bitsPerSample = 0;
    let dataSize = 0;
    let dataOffset = 0;

    while (offset < buffer.byteLength - 8) {
        const chunkId = String.fromCharCode(
            view.getUint8(offset),
            view.getUint8(offset + 1),
            view.getUint8(offset + 2),
            view.getUint8(offset + 3)
        );
        const chunkSize = view.getUint32(offset + 4, true);

        if (chunkId === "fmt ") {
            // fmt 子块
            audioFormat = view.getUint16(offset + 8, true);
            numChannels = view.getUint16(offset + 10, true);
            sampleRate = view.getUint32(offset + 12, true);
            byteRate = view.getUint32(offset + 16, true);
            blockAlign = view.getUint16(offset + 20, true);
            bitsPerSample = view.getUint16(offset + 22, true);
        } else if (chunkId === "data") {
            // data 子块
            dataSize = chunkSize;
            dataOffset = offset + 8;
            break; // 找到 data 后停止
        }

        // 跳到下一个子块
        offset += 8 + chunkSize;
        // 确保对齐到偶数字节
        if (chunkSize % 2 !== 0) {
            offset += 1;
        }
    }

    if (audioFormat === 0) {
        throw new Error("Invalid WAV file: fmt chunk not found");
    }
    if (dataOffset === 0) {
        throw new Error("Invalid WAV file: data chunk not found");
    }
    if (audioFormat !== 1) {
        throw new Error(`Unsupported audio format: ${audioFormat} (only PCM is supported)`);
    }

    return {
        audioFormat,
        numChannels,
        sampleRate,
        byteRate,
        blockAlign,
        bitsPerSample,
        dataSize,
        dataOffset,
    };
}

/**
 * 从 WAV 文件中提取 PCM 采样数据
 *
 * @param buffer - WAV 文件的 ArrayBuffer
 * @param header - WAV 头信息
 * @returns Float32Array 采样数据 (单声道, 范围 [-1, 1])
 */
export function extractPcmSamples(buffer: ArrayBuffer, header: WavHeader): Float32Array {
    const { numChannels, bitsPerSample, dataSize, dataOffset, blockAlign } = header;

    const bytesPerSample = bitsPerSample / 8;
    const numSamplesPerChannel = Math.floor(dataSize / blockAlign);

    // 输出单声道
    const samples = new Float32Array(numSamplesPerChannel);
    const view = new DataView(buffer);

    for (let i = 0; i < numSamplesPerChannel; i++) {
        let sum = 0;

        // 平均所有声道
        for (let ch = 0; ch < numChannels; ch++) {
            const sampleOffset = dataOffset + i * blockAlign + ch * bytesPerSample;

            let value: number;
            if (bitsPerSample === 8) {
                // 8-bit: unsigned, 0-255
                value = (view.getUint8(sampleOffset) - 128) / 128;
            } else if (bitsPerSample === 16) {
                // 16-bit: signed, -32768 to 32767
                value = view.getInt16(sampleOffset, true) / 32768;
            } else if (bitsPerSample === 24) {
                // 24-bit: signed
                const b0 = view.getUint8(sampleOffset);
                const b1 = view.getUint8(sampleOffset + 1);
                const b2 = view.getUint8(sampleOffset + 2);
                let val = (b2 << 16) | (b1 << 8) | b0;
                if (val >= 0x800000) {
                    val -= 0x1000000;
                }
                value = val / 8388608; // 2^23
            } else if (bitsPerSample === 32) {
                // 32-bit: 可能是 float 或 int
                // 这里假设是 32-bit int
                value = view.getInt32(sampleOffset, true) / 2147483648; // 2^31
            } else {
                throw new Error(`Unsupported bits per sample: ${bitsPerSample}`);
            }

            sum += value;
        }

        // 平均
        samples[i] = sum / numChannels;
    }

    return samples;
}

/**
 * 解析 WAV 文件
 *
 * @param filePath - WAV 文件路径
 * @returns 解析后的音频数据
 */
export async function parseWavFile(filePath: string): Promise<ParsedAudio> {
    const fileBuffer = await fs.readFile(filePath);
    const arrayBuffer = fileBuffer.buffer.slice(
        fileBuffer.byteOffset,
        fileBuffer.byteOffset + fileBuffer.byteLength
    );

    const header = parseWavHeader(arrayBuffer);
    const samples = extractPcmSamples(arrayBuffer, header);

    return { header, samples };
}

// ============================================================================
// Whisper 预处理
// ============================================================================

/**
 * Whisper Mel 频谱图预处理器
 *
 * 对标 OpenAI Whisper 的 log_mel_spectrogram 函数
 */
export class WhisperFeatureExtractor {
    private readonly sampleRate: number;
    private readonly nFft: number;
    private readonly hopLength: number;
    private readonly nMels: number;
    private readonly chunkLength: number;
    private readonly nSamples: number;

    // 重采样器
    private resamplers: Map<number, audio.Resample> = new Map();

    // MelSpectrogram 变换
    private melSpectrogram: audio.MelSpectrogram;

    constructor(nMels: number = 80) {
        this.sampleRate = WHISPER_AUDIO_CONFIG.SAMPLE_RATE;
        this.nFft = WHISPER_AUDIO_CONFIG.N_FFT;
        this.hopLength = WHISPER_AUDIO_CONFIG.HOP_LENGTH;
        this.nMels = nMels;
        this.chunkLength = WHISPER_AUDIO_CONFIG.CHUNK_LENGTH;
        this.nSamples = WHISPER_AUDIO_CONFIG.N_SAMPLES;

        // 创建 MelSpectrogram 变换
        // OpenAI Whisper 使用 librosa 的默认设置:
        // - mel_scale: 'slaney' (Slaney 公式的 Hz-Mel 转换)
        // - norm: 'slaney' (按带宽归一化)
        this.melSpectrogram = new audio.MelSpectrogram({
            sample_rate: this.sampleRate,
            n_fft: this.nFft,
            hop_length: this.hopLength,
            n_mels: nMels,
            power: 2.0, // 功率谱
            f_max: 8000, // Whisper 使用 fmax=8000
            norm: "slaney",
            mel_scale: "slaney",
        });
    }

    /**
     * 获取或创建重采样器
     */
    private getResampler(origFreq: number): audio.Resample {
        if (!this.resamplers.has(origFreq)) {
            this.resamplers.set(
                origFreq,
                new audio.Resample({
                    orig_freq: origFreq,
                    new_freq: this.sampleRate,
                })
            );
        }
        return this.resamplers.get(origFreq)!;
    }

    /**
     * 预处理音频
     *
     * 1. 重采样到 16kHz
     * 2. 填充或截断到 30 秒
     * 3. 计算 Mel 频谱图
     * 4. 应用 Whisper 特定的 log 归一化
     *
     * @param samples - 音频采样 (Float32Array, 单声道)
     * @param origSampleRate - 原始采样率
     * @returns Mel 频谱图张量，形状 (1, n_mels, n_frames)
     */
    async extract(samples: Float32Array, origSampleRate: number): Promise<Tensor> {
        // ==========================================
        // 1. 重采样
        // ==========================================
        let waveform = new Tensor(samples, { shape: [1, samples.length], dtype: "float32" });

        if (origSampleRate !== this.sampleRate) {
            const resampler = this.getResampler(origSampleRate);
            waveform = await resampler.forward(waveform);
        }

        // ==========================================
        // 2. 填充或截断到 30 秒
        // ==========================================
        const currentLength = waveform.shape[1];
        if (currentLength > this.nSamples) {
            // 截断
            waveform = waveform.slice(`:, 0:${this.nSamples}`);
        } else if (currentLength < this.nSamples) {
            // 填充 (补零)
            const padLength = this.nSamples - currentLength;
            const padded = new Float32Array(this.nSamples);
            const waveformData = (await waveform.dataAsync()) as Float32Array;
            padded.set(waveformData);
            waveform.dispose();
            waveform = new Tensor(padded, { shape: [1, this.nSamples], dtype: "float32" });
        }

        // ==========================================
        // 3. 计算 Mel 频谱图
        // ==========================================
        // MelSpectrogram 期望输入 (..., time)
        let melSpec = await this.melSpectrogram.forward(waveform);
        waveform.dispose();

        // melSpec 形状: (1, n_mels, n_frames)
        //
        // 重要: OpenAI Whisper 的实现中会丢弃最后一帧
        // magnitudes = stft[..., :-1].abs() ** 2
        // 这是因为 STFT with center=True 会产生 3001 帧，需要截断到 3000 帧
        const actualFrames = melSpec.shape[2];
        const expectedFrames = Math.floor(this.nSamples / this.hopLength); // 3000

        if (actualFrames > expectedFrames) {
            // 丢弃最后 (actualFrames - expectedFrames) 帧，与 OpenAI Whisper 对齐
            const sliced = melSpec.slice(`:, :, 0:${expectedFrames}`);
            melSpec.dispose();
            melSpec = sliced;
        }

        // ==========================================
        // 4. Whisper 特定的 log 归一化
        // ==========================================
        // log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        // log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        // log_spec = (log_spec + 4.0) / 4.0

        const melData = (await melSpec.dataAsync()) as Float32Array;
        melSpec.dispose();

        const logSpec = new Float32Array(melData.length);

        // Step 1: log10(clamp(x, 1e-10))
        let maxVal = -Infinity;
        for (let i = 0; i < melData.length; i++) {
            const clamped = Math.max(melData[i], 1e-10);
            logSpec[i] = Math.log10(clamped);
            if (logSpec[i] > maxVal) {
                maxVal = logSpec[i];
            }
        }

        // Step 2: max(log_spec, max - 8.0)
        const threshold = maxVal - 8.0;
        for (let i = 0; i < logSpec.length; i++) {
            if (logSpec[i] < threshold) {
                logSpec[i] = threshold;
            }
        }

        // Step 3: (log_spec + 4.0) / 4.0
        for (let i = 0; i < logSpec.length; i++) {
            logSpec[i] = (logSpec[i] + 4.0) / 4.0;
        }

        // 创建输出张量
        // 帧数已在步骤 3 中确保为 expectedFrames (3000)
        const nFrames = logSpec.length / this.nMels;
        return new Tensor(logSpec, {
            shape: [1, this.nMels, nFrames],
            dtype: "float32",
        });
    }

    /**
     * 从 WAV 文件提取特征
     *
     * @param filePath - WAV 文件路径
     * @returns Mel 频谱图张量，形状 (1, n_mels, n_frames)
     */
    async fromFile(filePath: string): Promise<Tensor> {
        const { header, samples } = await parseWavFile(filePath);
        return this.extract(samples, header.sampleRate);
    }
}
