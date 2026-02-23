/**
 * Hidden Markov Model (HMM) utilities implementing the Baum-Welch algorithm.
 * 
 * Notation:
 * N: Number of states
 * M: Number of distinct observations
 * T: Length of observation sequence
 * Pi: Initial probabilities [N]
 * A: Transition matrix [N x N]
 * B: Emission matrix [N x M]
 * O: Observation sequence [T] mapped to indices (0 to M-1)
 */

/**
 * Compute the Forward probabilities (\alpha)
 */
export function forward(O, Pi, A, B) {
    const T = O.length;
    const N = Pi.length;
    const alpha = Array.from({ length: T }, () => new Array(N).fill(0));

    if (T === 0) return { alpha, p: 0 };

    // Initialization: t = 0
    let p = 0;
    for (let i = 0; i < N; i++) {
        alpha[0][i] = Pi[i] * (B[i][O[0]] || 0);
    }

    // Recursion
    for (let t = 1; t < T; t++) {
        for (let j = 0; j < N; j++) {
            let sum = 0;
            for (let i = 0; i < N; i++) {
                sum += alpha[t - 1][i] * A[i][j];
            }
            alpha[t][j] = sum * (B[j][O[t]] || 0);
        }
    }

    // Probability of sequence
    p = alpha[T - 1].reduce((sum, val) => sum + val, 0);

    return { alpha, p };
}

/**
 * Compute the Backward probabilities (\beta)
 */
export function backward(O, Pi, A, B) {
    const T = O.length;
    const N = Pi.length;
    const beta = Array.from({ length: T }, () => new Array(N).fill(0));

    if (T === 0) return beta;

    // Initialization: t = T - 1
    for (let i = 0; i < N; i++) {
        beta[T - 1][i] = 1;
    }

    // Recursion
    for (let t = T - 2; t >= 0; t--) {
        for (let i = 0; i < N; i++) {
            let sum = 0;
            for (let j = 0; j < N; j++) {
                sum += A[i][j] * (B[j][O[t + 1]] || 0) * beta[t + 1][j];
            }
            beta[t][i] = sum;
        }
    }

    return beta;
}

/**
 * Compute State Responsibilities (\gamma) and Transition Responsibilities (\xi)
 */
export function computeResponsibilities(O, Pi, A, B, alpha, beta, p) {
    const T = O.length;
    const N = Pi.length;

    const gamma = Array.from({ length: T }, () => new Array(N).fill(0));
    const xi = Array.from({ length: T - 1 }, () =>
        Array.from({ length: N }, () => new Array(N).fill(0))
    );

    for (let t = 0; t < T; t++) {
        // Normalization factor exactly equal to p, but sometimes p->0 causes issues,
        // so we re-normalize row by row to prevent NaN if sequence is very unlikely.
        let rowSum = 0;
        for (let i = 0; i < N; i++) {
            gamma[t][i] = alpha[t][i] * beta[t][i];
            rowSum += gamma[t][i];
        }

        // Normalize gamma (to help numerical stability if p is extremely small)
        if (rowSum > 0) {
            for (let i = 0; i < N; i++) gamma[t][i] /= rowSum;
        }

        if (t < T - 1) {
            let xiSum = 0;
            for (let i = 0; i < N; i++) {
                for (let j = 0; j < N; j++) {
                    xi[t][i][j] = alpha[t][i] * A[i][j] * (B[j][O[t + 1]] || 0) * beta[t + 1][j];
                    xiSum += xi[t][i][j];
                }
            }
            // Normalize xi
            if (xiSum > 0) {
                for (let i = 0; i < N; i++) {
                    for (let j = 0; j < N; j++) {
                        xi[t][i][j] /= xiSum;
                    }
                }
            }
        }
    }

    return { gamma, xi };
}

/**
 * One iteration of the Baum-Welch update (M-step)
 */
export function baumWelchUpdate(O, Pi, A, B, gamma, xi, M) {
    const T = O.length;
    const N = Pi.length;

    // New parameters
    const newPi = [...Pi];
    const newA = A.map(row => [...row]);
    const newB = B.map(row => [...row]);

    // Update Pi
    for (let i = 0; i < N; i++) {
        newPi[i] = gamma[0][i];
    }

    // Update A
    for (let i = 0; i < N; i++) {
        let denom = 0;
        for (let t = 0; t < T - 1; t++) {
            denom += gamma[t][i];
        }

        if (denom > 0) {
            for (let j = 0; j < N; j++) {
                let num = 0;
                for (let t = 0; t < T - 1; t++) {
                    num += xi[t][i][j];
                }
                newA[i][j] = num / denom;
            }
        }
    }

    // Update B
    for (let i = 0; i < N; i++) {
        let denom = 0;
        for (let t = 0; t < T; t++) {
            denom += gamma[t][i];
        }

        if (denom > 0) {
            for (let k = 0; k < M; k++) {
                let num = 0;
                for (let t = 0; t < T; t++) {
                    if (O[t] === k) {
                        num += gamma[t][i];
                    }
                }
                newB[i][k] = num / denom;
            }
        }
    }

    return { Pi: newPi, A: newA, B: newB };
}

/**
 * Helper to run the entire alg for one iteration easily from UI
 */
export function baumWelchStep(O_str_array, obs_map, Pi, A, B) {
    // Map observations to indices
    const O = O_str_array.map(o => obs_map.indexOf(o)).filter(i => i !== -1);
    const M_obs = obs_map.length;

    if (O.length === 0) {
        return { Pi, A, B, alpha: [], beta: [], gamma: [], xi: [], p: 0, error: "Empty valid observation sequence." };
    }

    const { alpha, p } = forward(O, Pi, A, B);
    const beta = backward(O, Pi, A, B);
    const { gamma, xi } = computeResponsibilities(O, Pi, A, B, alpha, beta, p);
    const nextParams = baumWelchUpdate(O, Pi, A, B, gamma, xi, M_obs);

    return {
        ...nextParams, // Pi, A, B
        alpha,
        beta,
        gamma,
        xi,
        p
    };
}
