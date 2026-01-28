import { ITensorHandle } from "./tensor";

export interface IAudio {
    load(filePath: string | ArrayBuffer, sampleRate?: number): ITensorHandle;
    save(filePath: string, audioTensor: ITensorHandle, sampleRate: number): void;
    resample(a: ITensorHandle, origFreq: number, newFreq: number): ITensorHandle;

    /**
     * Create a spectrogram from an audio signal.
     * Returns the magnitude of the STFT.
     */
    spectrogram(a: ITensorHandle, nFFT: number, hopLength: number, winLength: number): ITensorHandle;

    melSpectrogram(
        a: ITensorHandle,
        sampleRate: number,
        nFFT: number,
        hopLength: number,
        winLength: number,
        nMels: number
    ): ITensorHandle;
    mfcc(
        a: ITensorHandle,
        sampleRate: number,
        nFFT: number,
        hopLength: number,
        winLength: number,
        nMels: number,
        nMFCC: number
    ): ITensorHandle;
}
