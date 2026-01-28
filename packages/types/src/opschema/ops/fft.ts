/**
 * FFT Operations Schema
 * 
 * Defines FFT-related operations for the fft.* namespace
 */

import { SchemaT, SchemaShape, SchemaDtype } from '../helpers';
import type { OpEntry } from '../types';

/**
 * torch.fft.fft - 1D Discrete Fourier Transform
 * 
 * Computes the one dimensional discrete Fourier transform of input.
 */
export const fft: OpEntry = {
    name: 'fft',
    mechanism: 'FFT', // FFT mechanism
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor' }, doc: 'Input tensor (real or complex)' },
            { name: 'n', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Signal length. If None, uses input size along dim' },
            { name: 'dim', type: { kind: 'Scalar', numericKind: 'int' }, default: -1, doc: 'Dimension to transform' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'same', as: 'input' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'fft',
    doc: 'Computes the one dimensional discrete Fourier transform of input.',
    codegen: {
        namespace: 'fft',  // Generates in fft.* namespace
        namespaceKeyAlias: 'fft',  // Function becomes fftImpl, exported as fft.fft
        tensorMethod: false,
    },
};

/**
 * torch.fft.ifft - 1D Inverse Discrete Fourier Transform
 * 
 * Computes the one dimensional inverse discrete Fourier transform of input.
 */
export const ifft: OpEntry = {
    name: 'ifft',
    mechanism: 'FFT',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Complex' }, doc: 'Input tensor (complex)' },
            { name: 'n', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Signal length' },
            { name: 'dim', type: { kind: 'Scalar', numericKind: 'int' }, default: -1, doc: 'Dimension to transform' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'same', as: 'input' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'ifft',
    doc: 'Computes the one dimensional inverse discrete Fourier transform of input.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.rfft - 1D Real FFT
 * 
 * Computes the one dimensional FFT of real-valued input.
 * Output has length n//2 + 1 (only positive frequencies).
 */
export const rfft: OpEntry = {
    name: 'rfft',
    mechanism: 'FFT',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Floating' }, doc: 'Real input tensor' },
            { name: 'n', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Signal length' },
            { name: 'dim', type: { kind: 'Scalar', numericKind: 'int' }, default: -1, doc: 'Dimension to transform' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'explicit', expr: 'input.shape with dim reduced to n//2+1' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'rfft',
    doc: 'Computes the one dimensional FFT of real-valued input.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.irfft - 1D Inverse Real FFT
 * 
 * Computes the inverse FFT of rfft. Output is real-valued.
 */
export const irfft: OpEntry = {
    name: 'irfft',
    mechanism: 'FFT',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Complex' }, doc: 'Complex input tensor' },
            { name: 'n', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Output signal length' },
            { name: 'dim', type: { kind: 'Scalar', numericKind: 'int' }, default: -1, doc: 'Dimension to transform' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'float32' } },
    },
    shape: { rule: 'explicit', expr: 'input.shape with dim expanded to n' },
    dtype: { rule: 'fixed', dtype: 'float32' },
    dispatchKey: 'irfft',
    doc: 'Computes the inverse FFT of rfft. Output is real-valued.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.fftshift - Shift zero-frequency component to center
 * 
 * Rearranges the FFT output so that the zero-frequency component is at the center.
 * For dim of length n: [0, 1, ..., n//2-1, n//2, ..., n-1] -> [n//2, ..., n-1, 0, 1, ..., n//2-1]
 */
export const fftshift: OpEntry = {
    name: 'fftshift',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor(), doc: 'Input tensor' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axes()), doc: 'Dimensions to shift. Default: all dimensions' },
        ],
        returns: { single: { kind: 'Tensor' } },
    },
    shape: { rule: 'same', as: 'input' },
    dtype: { rule: 'same', as: 'input' },
    dispatchKey: 'fftshift',
    doc: 'Shift zero-frequency component to center of spectrum.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.ifftshift - Inverse of fftshift
 * 
 * Undoes the effect of fftshift.
 */
export const ifftshift: OpEntry = {
    name: 'ifftshift',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: SchemaT.Tensor(), doc: 'Input tensor' },
            { name: 'dim', type: SchemaT.Optional(SchemaT.Axes()), doc: 'Dimensions to shift. Default: all dimensions' },
        ],
        returns: { single: { kind: 'Tensor' } },
    },
    shape: { rule: 'same', as: 'input' },
    dtype: { rule: 'same', as: 'input' },
    dispatchKey: 'ifftshift',
    doc: 'Inverse of fftshift.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.fftfreq - DFT sample frequencies
 * 
 * Returns the sample frequencies for a signal of size n.
 * Output: [0, 1, ..., n//2-1, -n//2, ..., -1] / (n*d)
 */
export const fftfreq: OpEntry = {
    name: 'fftfreq',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'n', type: { kind: 'Scalar', numericKind: 'int' }, doc: 'Signal length' },
            { name: 'd', type: { kind: 'Scalar', numericKind: 'float' }, default: 1.0, doc: 'Sample spacing' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'float32' } },
    },
    shape: { rule: 'explicit', expr: '[n]' },
    dtype: { rule: 'fixed', dtype: 'float32' },
    dispatchKey: 'fftfreq',
    doc: 'Returns the DFT sample frequencies.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.rfftfreq - RFFT sample frequencies
 * 
 * Returns the sample frequencies for rfft output.
 * Output: [0, 1, ..., n//2] / (n*d) with length n//2+1
 */
export const rfftfreq: OpEntry = {
    name: 'rfftfreq',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'n', type: { kind: 'Scalar', numericKind: 'int' }, doc: 'Signal length (of the original real signal)' },
            { name: 'd', type: { kind: 'Scalar', numericKind: 'float' }, default: 1.0, doc: 'Sample spacing' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'float32' } },
    },
    shape: { rule: 'explicit', expr: '[n//2+1]' },
    dtype: { rule: 'fixed', dtype: 'float32' },
    dispatchKey: 'rfftfreq',
    doc: 'Returns the sample frequencies for rfft.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.stft - Short-Time Fourier Transform
 * 
 * Computes the Short-Time Fourier Transform (STFT) of a signal.
 * The signal is divided into short overlapping segments, and the Fourier transform
 * is computed for each segment.
 * 
 * Note: This is a top-level function, NOT in the fft namespace (matching PyTorch).
 */
export const stft: OpEntry = {
    name: 'stft',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Floating' }, doc: 'Input signal tensor (B?, L)' },
            { name: 'n_fft', type: { kind: 'Scalar', numericKind: 'int' }, doc: 'Size of FFT window' },
            { name: 'hop_length', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Stride between STFT windows. Default: n_fft // 4' },
            { name: 'win_length', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Window size. Default: n_fft' },
            { name: 'window', type: { kind: 'Optional', inner: { kind: 'Tensor' } }, doc: 'Window function tensor' },
            { name: 'center', type: { kind: 'Bool' }, default: true, doc: 'Whether to pad input on both sides' },
            { name: 'pad_mode', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['constant', 'reflect', 'replicate', 'circular'] } }, doc: 'Padding mode. Default: reflect' },
            { name: 'normalized', type: { kind: 'Bool' }, default: false, doc: 'Whether to normalize by 1/sqrt(n_fft)' },
            { name: 'onesided', type: { kind: 'Optional', inner: { kind: 'Bool' } }, doc: 'Whether to return onesided output (default: true for real input)' },
            { name: 'return_complex', type: { kind: 'Bool' }, default: true, doc: 'Whether to return complex tensor' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'explicit', expr: '(B?, freq_bins, n_frames)' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'stft',
    doc: 'Computes the Short-Time Fourier Transform of the input signal.',
    codegen: {
        namespace: undefined,  // Top-level function, not in fft namespace
        tensorMethod: false,
    },
};

/**
 * torch.istft - Inverse Short-Time Fourier Transform
 * 
 * Computes the inverse Short-Time Fourier Transform (iSTFT) to reconstruct
 * a signal from its STFT representation.
 * 
 * Note: This is a top-level function, NOT in the fft namespace (matching PyTorch).
 */
export const istft: OpEntry = {
    name: 'istft',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Complex' }, doc: 'STFT tensor (B?, freq_bins, n_frames)' },
            { name: 'n_fft', type: { kind: 'Scalar', numericKind: 'int' }, doc: 'Size of FFT window' },
            { name: 'hop_length', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Stride between STFT windows. Default: n_fft // 4' },
            { name: 'win_length', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Window size. Default: n_fft' },
            { name: 'window', type: { kind: 'Optional', inner: { kind: 'Tensor' } }, doc: 'Window function tensor (must match stft window)' },
            { name: 'center', type: { kind: 'Bool' }, default: true, doc: 'Whether input was padded on both sides' },
            { name: 'normalized', type: { kind: 'Bool' }, default: false, doc: 'Whether STFT was normalized' },
            { name: 'onesided', type: { kind: 'Optional', inner: { kind: 'Bool' } }, doc: 'Whether input is onesided' },
            { name: 'length', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Expected output length' },
            { name: 'return_complex', type: { kind: 'Bool' }, default: false, doc: 'Whether to return complex output' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'float32' } },
    },
    shape: { rule: 'explicit', expr: '(B?, L)' },
    dtype: { rule: 'fixed', dtype: 'float32' },
    dispatchKey: 'istft',
    doc: 'Computes the inverse Short-Time Fourier Transform to reconstruct the signal.',
    codegen: {
        namespace: undefined,  // Top-level function, not in fft namespace
        tensorMethod: false,
    },
};

/**
 * torch.fft.fft2 - 2D Discrete Fourier Transform
 * 
 * Computes the 2-dimensional discrete Fourier transform of input.
 * Equivalent to fftn(input, s, dim) with dim=[-2, -1].
 */
export const fft2: OpEntry = {
    name: 'fft2',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor' }, doc: 'Input tensor' },
            { name: 's', type: { kind: 'Optional', inner: { kind: 'Shape' } }, doc: 'Signal sizes for last 2 dims' },
            { name: 'dim', type: { kind: 'Optional', inner: { kind: 'Axes' } }, doc: 'Dimensions to transform (default: [-2, -1])' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'same', as: 'input' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'fft2',
    doc: 'Computes the 2-dimensional discrete Fourier transform of input.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.ifft2 - 2D Inverse Discrete Fourier Transform
 * 
 * Computes the 2-dimensional inverse discrete Fourier transform of input.
 */
export const ifft2: OpEntry = {
    name: 'ifft2',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Complex' }, doc: 'Complex input tensor' },
            { name: 's', type: { kind: 'Optional', inner: { kind: 'Shape' } }, doc: 'Signal sizes for last 2 dims' },
            { name: 'dim', type: { kind: 'Optional', inner: { kind: 'Axes' } }, doc: 'Dimensions to transform (default: [-2, -1])' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'same', as: 'input' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'ifft2',
    doc: 'Computes the 2-dimensional inverse discrete Fourier transform of input.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.rfft2 - 2D Real FFT
 * 
 * Computes the 2-dimensional FFT of real-valued input.
 * The last dimension has length s[-1]//2 + 1 (only positive frequencies).
 */
export const rfft2: OpEntry = {
    name: 'rfft2',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Floating' }, doc: 'Real input tensor' },
            { name: 's', type: { kind: 'Optional', inner: { kind: 'Shape' } }, doc: 'Signal sizes for last 2 dims' },
            { name: 'dim', type: { kind: 'Optional', inner: { kind: 'Axes' } }, doc: 'Dimensions to transform (default: [-2, -1])' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'explicit', expr: 'input.shape with dim[-1] reduced to s[-1]//2+1' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'rfft2',
    doc: 'Computes the 2-dimensional FFT of real-valued input.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.irfft2 - 2D Inverse Real FFT
 * 
 * Computes the inverse 2D FFT of rfft2 output. Output is real-valued.
 */
export const irfft2: OpEntry = {
    name: 'irfft2',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Complex' }, doc: 'Complex input tensor' },
            { name: 's', type: { kind: 'Optional', inner: { kind: 'Shape' } }, doc: 'Output signal sizes for last 2 dims' },
            { name: 'dim', type: { kind: 'Optional', inner: { kind: 'Axes' } }, doc: 'Dimensions to transform (default: [-2, -1])' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'float32' } },
    },
    shape: { rule: 'explicit', expr: 'input.shape with dim[-1] expanded to s[-1]' },
    dtype: { rule: 'fixed', dtype: 'float32' },
    dispatchKey: 'irfft2',
    doc: 'Computes the inverse 2D FFT of rfft2 output. Output is real-valued.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};


/**
 * torch.fft.fftn - N-Dimensional Discrete Fourier Transform
 * 
 * Computes the N-dimensional discrete Fourier transform of input.
 */
export const fftn: OpEntry = {
    name: 'fftn',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor' }, doc: 'Input tensor' },
            { name: 's', type: { kind: 'Optional', inner: { kind: 'Shape' } }, doc: 'Signal size in the transformed dimensions' },
            { name: 'dim', type: { kind: 'Optional', inner: { kind: 'Axes' } }, doc: 'Dimensions to transform (default: all)' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'same', as: 'input' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'fftn',
    doc: 'Computes the N-dimensional discrete Fourier transform of input.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.ifftn - N-Dimensional Inverse Discrete Fourier Transform
 * 
 * Computes the N-dimensional inverse discrete Fourier transform of input.
 */
export const ifftn: OpEntry = {
    name: 'ifftn',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Complex' }, doc: 'Complex input tensor' },
            { name: 's', type: { kind: 'Optional', inner: { kind: 'Shape' } }, doc: 'Signal size in the transformed dimensions' },
            { name: 'dim', type: { kind: 'Optional', inner: { kind: 'Axes' } }, doc: 'Dimensions to transform (default: all)' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'same', as: 'input' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'ifftn',
    doc: 'Computes the N-dimensional inverse discrete Fourier transform of input.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.rfftn - N-Dimensional Real FFT
 * 
 * Computes the N-dimensional FFT of real-valued input.
 */
export const rfftn: OpEntry = {
    name: 'rfftn',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Floating' }, doc: 'Real input tensor' },
            { name: 's', type: { kind: 'Optional', inner: { kind: 'Shape' } }, doc: 'Signal size in the transformed dimensions' },
            { name: 'dim', type: { kind: 'Optional', inner: { kind: 'Axes' } }, doc: 'Dimensions to transform (default: all)' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'explicit', expr: 'input.shape with last dim reduced' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'rfftn',
    doc: 'Computes the N-dimensional FFT of real-valued input.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.irfftn - N-Dimensional Inverse Real FFT
 * 
 * Computes the N-dimensional inverse FFT of real input.
 */
export const irfftn: OpEntry = {
    name: 'irfftn',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Complex' }, doc: 'Complex input tensor' },
            { name: 's', type: { kind: 'Optional', inner: { kind: 'Shape' } }, doc: 'Output signal size in the transformed dimensions' },
            { name: 'dim', type: { kind: 'Optional', inner: { kind: 'Axes' } }, doc: 'Dimensions to transform (default: all)' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'float32' } },
    },
    shape: { rule: 'explicit', expr: 'input.shape with last dim expanded' },
    dtype: { rule: 'fixed', dtype: 'float32' },
    dispatchKey: 'irfftn',
    doc: 'Computes the N-dimensional inverse FFT of real input.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.hfft - 1D Hermitian FFT
 * 
 * Computes the one dimensional discrete Fourier transform of a Hermitian symmetric input signal.
 * The input represents a half-Hermitian signal in the time domain, and the output is real-valued
 * in the frequency domain.
 * 
 * This is functionally equivalent to: irfft(conj(input), n, dim, norm)
 */
export const hfft: OpEntry = {
    name: 'hfft',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Complex' }, doc: 'Half-Hermitian complex input tensor' },
            { name: 'n', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Output signal length. Default: 2*(input.size(dim)-1)' },
            { name: 'dim', type: { kind: 'Scalar', numericKind: 'int' }, default: -1, doc: 'Dimension to transform' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'float32' } },
    },
    shape: { rule: 'explicit', expr: 'input.shape with dim expanded to n' },
    dtype: { rule: 'fixed', dtype: 'float32' },
    dispatchKey: 'hfft',
    doc: 'Computes the 1D discrete Fourier transform of a Hermitian symmetric input signal.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};

/**
 * torch.fft.ihfft - 1D Inverse Hermitian FFT
 * 
 * Computes the inverse of hfft. The input must be a real-valued signal, interpreted in the
 * Fourier domain. The IFFT of such a real signal is Hermitian-symmetric. ihfft represents
 * this in a one-sided form where only positive frequencies below the Nyquist frequency
 * are included.
 * 
 * This is functionally equivalent to: conj(rfft(input, n, dim, norm))
 */
export const ihfft: OpEntry = {
    name: 'ihfft',
    mechanism: 'Composite',
    signature: {
        params: [
            { name: 'input', type: { kind: 'Tensor', dtype: 'Floating' }, doc: 'Real input tensor' },
            { name: 'n', type: { kind: 'Optional', inner: { kind: 'Scalar', numericKind: 'int' } }, doc: 'Signal length' },
            { name: 'dim', type: { kind: 'Scalar', numericKind: 'int' }, default: -1, doc: 'Dimension to transform' },
            { name: 'norm', type: { kind: 'Optional', inner: { kind: 'String', oneOf: ['forward', 'backward', 'ortho'] } }, doc: 'Normalization mode' },
        ],
        returns: { single: { kind: 'Tensor', dtype: 'complex64' } },
    },
    shape: { rule: 'explicit', expr: 'Half-length output: n//2 + 1' },
    dtype: { rule: 'fixed', dtype: 'complex64' },
    dispatchKey: 'ihfft',
    doc: 'Computes the inverse of hfft.',
    codegen: {
        namespace: 'fft',
        tensorMethod: false,
    },
};
