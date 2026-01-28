/**
 * Twiddle Factor Precomputation
 * 
 * Precomputes the rotation factors (twiddle factors) for FFT:
 * W_N^k = exp(±2πik/N) = cos(2πk/N) ± i·sin(2πk/N)
 */

// Cache for twiddle factor buffers: Map<size, GPUBuffer>
const twiddleBufferCache = new Map<number, GPUBuffer>();

/**
 * Get or create a GPU buffer containing twiddle factors for FFT size N.
 * 
 * The buffer contains Nval entries (where Nval = N/2) of Complex numbers.
 * Each entry k corresponds to W_N^k = exp(-2πik/N).
 * Layout: [re, im, re, im, ...] (float32)
 * 
 * We store the "forward" factors (negative exponent).
 * For inverse FFT, the shader can adjust (conjugate or sign flip).
 * 
 * Size of buffer: (N/2) * 2 * 4 bytes = 4N bytes.
 * 
 * @param device GPUDevice to create buffer on
 * @param n FFT size (must be power of 2)
 */
export function getTwiddleBuffer(device: GPUDevice, n: number): GPUBuffer {
    if (twiddleBufferCache.has(n)) {
        return twiddleBufferCache.get(n)!;
    }

    // Compute twiddle factors on CPU using float64
    // We only need k = 0 to N/2 - 1
    const halfN = n / 2;
    const data = new Float32Array(halfN * 2);

    for (let k = 0; k < halfN; k++) {
        // Forward transform angle: -2πk/N
        // We store cos(2πk/N) and sin(2πk/N) and let shader handle sign?
        // Let's store exp(-2πik/N) explicitly: cos(-x) + i sin(-x)
        // = cos(2πk/N) - i sin(2πk/N)
        // Real part: cos, Imag part: -sin
        const angle = -2 * Math.PI * k / n;
        data[k * 2] = Math.cos(angle);
        data[k * 2 + 1] = Math.sin(angle);
    }

    // Create GPU buffer
    // Usage: STORAGE (read-only in shader), COPY_DST (for upload)
    const buffer = device.createBuffer({
        label: `fft_twiddle_${n}`,
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });

    // Write data
    new Float32Array(buffer.getMappedRange()).set(data);
    buffer.unmap();

    // Cache it
    twiddleBufferCache.set(n, buffer);

    return buffer;
}

/**
 * Clear the twiddle factor cache (e.g. on device loss)
 */
export function clearTwiddleCache() {
    for (const buffer of twiddleBufferCache.values()) {
        buffer.destroy();
    }
    twiddleBufferCache.clear();
}

/**
 * Compute twiddle factors for FFT of size N (Legacy/CPU usage)
 * 
 * @param n FFT size (must be power of 2)
 * @param direction 'forward' uses -2πi, 'inverse' uses +2πi
 * @returns Float32Array of [real, imag] pairs for k = 0 to n-1
 */
export function computeTwiddleFactors(
    n: number,
    direction: 'forward' | 'inverse'
): Float32Array {
    const sign = direction === 'forward' ? -1 : 1;
    const factors = new Float32Array(n * 2); // [real, imag] pairs

    for (let k = 0; k < n; k++) {
        const angle = sign * 2 * Math.PI * k / n;
        factors[k * 2] = Math.cos(angle);     // real
        factors[k * 2 + 1] = Math.sin(angle); // imag
    }

    return factors;
}

/**
 * Compute twiddle factors for a specific FFT stage (Legacy)
 */
export function computeStageTwiddleFactors(
    stage: number,
    direction: 'forward' | 'inverse'
): Float32Array {
    const blockSize = 1 << (stage + 1);
    const halfBlock = blockSize >> 1;
    const sign = direction === 'forward' ? -1 : 1;
    const factors = new Float32Array(halfBlock * 2);

    for (let k = 0; k < halfBlock; k++) {
        const angle = sign * 2 * Math.PI * k / blockSize;
        factors[k * 2] = Math.cos(angle);
        factors[k * 2 + 1] = Math.sin(angle);
    }

    return factors;
}
