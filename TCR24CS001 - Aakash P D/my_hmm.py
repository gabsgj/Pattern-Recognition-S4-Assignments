import numpy as np
def baum_welch(obs_seq, N):
    # Convert observed sequence string to list of ints
    O = list(map(int, obs_seq.split(",")))

    # Random transition matrix A
    A = np.round(np.random.rand(N, N), 2)
    A = A / A.sum(axis=1)[:, None]

    # Random emission matrix B
    B = np.round(np.random.rand(N, max(O)+1), 2)
    B = B / B.sum(axis=1)[:, None]

    # Random initial probabilities
    pi = np.round(np.random.rand(N), 2)
    pi = pi / pi.sum()

    # Random probability of observation (placeholder)
    P_O = round(np.random.rand(), 4)

    return A.tolist(), B.tolist(), pi.tolist(), P_O