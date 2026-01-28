/**
 * Mathematical Utility Functions
 * 
 * Provides cross-platform mathematical computation tools for frontend and backend.
 */

/**
 * Compute the greatest common divisor (GCD) of two integers using Euclidean algorithm.
 * 
 * @param a First integer
 * @param b Second integer
 * @returns GCD(a, b)
 */
export function gcd(a: number, b: number): number {
    a = Math.abs(Math.floor(a));
    b = Math.abs(Math.floor(b));
    while (b !== 0) {
        const temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

/**
 * Compute the modified Bessel function of the first kind of order zero: I₀(x)
 * 
 * Uses rational approximation from Abramowitz & Stegun.
 * Precision: Relative error < 1.6e-7
 * 
 * Reference:
 * - Abramowitz, M., & Stegun, I. A. (1964). Handbook of Mathematical Functions.
 * - Chapter 9: Bessel Functions of Integer Order
 * 
 * @param x Input value
 * @returns I₀(x)
 */
export function computeBesselI0(x: number): number {
    const ax = Math.abs(x);
    let ans: number;

    if (ax < 3.75) {
        // Polynomial approximation for |x| < 3.75
        const y = (x / 3.75) * (x / 3.75);
        ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
            + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
    } else {
        // Asymptotic expansion for |x| >= 3.75
        const y = 3.75 / ax;
        ans = (Math.exp(ax) / Math.sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
            + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
                + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
                    + y * 0.392377e-2))))))));
    }
    return ans;
}
