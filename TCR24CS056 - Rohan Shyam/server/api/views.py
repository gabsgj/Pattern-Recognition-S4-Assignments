# server/api/views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
import numpy as np

@api_view(['POST'])
def run_baum_welch(req):
    data = req.data 
    
    # 1. Extract and format the incoming data
    n_iter = int(data.get('iterations', 100))
    O = np.array(data.get('sequence'), dtype=int)
    
    # Convert incoming JSON arrays to NumPy matrices for fast math
    A = np.array(data.get('matrix_a'), dtype=float)
    B = np.array(data.get('matrix_b'), dtype=float)
    pi = np.array(data.get('pi'), dtype=float)
    
    N = A.shape[0] # Number of hidden states
    T = len(O)     # Length of observation sequence
    M = B.shape[1] # Number of possible observation symbols
    
    # This list will hold the snapshot of every single iteration for your React tables
    iteration_history = []
    
    # 2. The Baum-Welch Loop
    for it in range(n_iter):
        
        # --- EXPECTATION STEP ---
        
        # Forward Procedure (Alpha)
        alpha = np.zeros((T, N))
        alpha[0, :] = pi * B[:, O[0]]
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t-1, :] * A[:, j]) * B[j, O[t]]
                
        # Backward Procedure (Beta)
        beta = np.zeros((T, N))
        beta[T-1, :] = 1.0
        for t in range(T-2, -1, -1):
            for i in range(N):
                beta[t, i] = np.sum(A[i, :] * B[:, O[t+1]] * beta[t+1, :])
                
        # Calculate Gamma (Probability of being in state i at time t)
        gamma = np.zeros((T, N))
        for t in range(T):
            denominator = np.sum(alpha[t, :] * beta[t, :])
            # Prevent division by zero if probabilities get too small
            if denominator == 0: denominator = 1e-12 
            gamma[t, :] = (alpha[t, :] * beta[t, :]) / denominator
            
        # Calculate Xi (Probability of being in state i at t, and state j at t+1)
        xi = np.zeros((T-1, N, N))
        for t in range(T-1):
            denominator = np.sum(alpha[t, :, np.newaxis] * A * B[:, O[t+1]] * beta[t+1, :])
            if denominator == 0: denominator = 1e-12
            for i in range(N):
                for j in range(N):
                    xi[t, i, j] = (alpha[t, i] * A[i, j] * B[j, O[t+1]] * beta[t+1, j]) / denominator

        # Calculate current probability of the sequence P(O|λ)
        current_probability = np.sum(alpha[T-1, :])
        
        # Save snapshot for the React frontend BEFORE updating
        # (We use .tolist() because Django cannot send raw NumPy arrays over JSON)
        iteration_history.append({
            "iteration": it + 1,
            "matrix_a": A.tolist(),
            "matrix_b": B.tolist(),
            "pi": pi.tolist(),
            "probability": current_probability
        })

        # --- MAXIMIZATION STEP ---
        
        # Update Initial Probabilities (π)
        pi = gamma[0, :]
        
        # Update Transition Matrix (A)
        gamma_sum_A = np.sum(gamma[:-1, :], axis=0)
        gamma_sum_A[gamma_sum_A == 0] = 1e-12 # Safety check
        A = np.sum(xi, axis=0) / gamma_sum_A[:, np.newaxis]
        
        # Update Emission Matrix (B)
        gamma_sum_B = np.sum(gamma, axis=0)
        gamma_sum_B[gamma_sum_B == 0] = 1e-12 # Safety check
        for k in range(M):
            mask = (O == k)
            B[:, k] = np.sum(gamma[mask, :], axis=0) / gamma_sum_B

    # 3. Send data back to Next.js
    return Response({
        "status": "success",
        "message": f"Algorithm completed {n_iter} iterations.",
        "final_probability": current_probability,
        "final_matrix_a": A.tolist(),
        "final_matrix_b": B.tolist(),
        "final_pi": pi.tolist(),
        "history": iteration_history # Your React tables will use this!
    })