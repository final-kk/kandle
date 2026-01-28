/**
 * Random State Management
 * 
 * Manages the global PRNG state for random number generation.
 * This is placed in @kandle/types so that backends can access it.
 * 
 * PyTorch reference:
 * - torch.manual_seed(seed) → 设置种子
 * - torch.initial_seed() → 获取种子
 */

/**
 * Singleton class managing global random state
 * 
 * Uses Philox 4x32-10 algorithm internally:
 * - 64-bit seed → Philox key
 * - Global offset counter → ensures different calls produce different sequences
 */
export class RandomState {
    private static instance: RandomState | null = null;

    /** 64-bit seed (stored as BigInt for precision) */
    private seed: bigint = 0n;

    /** Global offset counter (incremented after each call) */
    private offset: number = 0;

    private constructor() {
        // Default seed: use current timestamp
        this.seed = BigInt(Date.now());
    }

    /**
     * Get the singleton instance
     */
    static getInstance(): RandomState {
        if (!RandomState.instance) {
            RandomState.instance = new RandomState();
        }
        return RandomState.instance;
    }

    /**
     * Reset instance (for testing purposes only)
     * @internal
     */
    static resetInstance(): void {
        RandomState.instance = null;
    }

    /**
     * Set the random seed
     * Aligns with torch.manual_seed(seed)
     * 
     * @param seed - The seed value (number or bigint)
     */
    setSeed(seed: number | bigint): void {
        this.seed = BigInt(seed);
        this.offset = 0; // Reset offset when seed changes
    }

    /**
     * Get the current seed
     * Aligns with torch.initial_seed()
     */
    getSeed(): bigint {
        return this.seed;
    }

    /**
     * Get Philox key as two 32-bit integers
     * 
     * Philox uses a 64-bit key, split into:
     * - key0: lower 32 bits
     * - key1: upper 32 bits
     */
    getKey(): [number, number] {
        const key0 = Number(this.seed & 0xFFFFFFFFn);
        const key1 = Number((this.seed >> 32n) & 0xFFFFFFFFn);
        return [key0, key1];
    }

    /**
     * Consume offset and return the current value
     * 
     * This ensures different random operations produce different sequences
     * even with the same seed.
     * 
     * @param count - Number of Philox calls consumed (numel / 4)
     * @returns The offset before consumption
     */
    consumeOffset(count: number): number {
        const current = this.offset;
        this.offset += count;
        return current;
    }

    /**
     * Get current offset without consuming
     */
    getOffset(): number {
        return this.offset;
    }
}

// ========================================
// Public API Functions
// ========================================

/**
 * Set the random seed for reproducibility
 * Aligns with torch.manual_seed(seed)
 * 
 * @example
 * ```ts
 * import * as torch from '@kandle/core';
 * 
 * torch.manualSeed(42);
 * const a = torch.rand([2, 2]);
 * 
 * torch.manualSeed(42);
 * const b = torch.rand([2, 2]);
 * // a and b will have identical values
 * ```
 */
export function manualSeed(seed: number | bigint): void {
    RandomState.getInstance().setSeed(seed);
}

/**
 * Get the initial seed value
 * Aligns with torch.initial_seed()
 */
export function initialSeed(): bigint {
    return RandomState.getInstance().getSeed();
}
