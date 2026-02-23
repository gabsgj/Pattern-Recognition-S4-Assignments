import streamlit as st
import numpy as np
import pandas as pd 

st.title("Baum-Welch Algorithm Visualizer")

states = ['Rainy','Sunny']
observations = ['Walk','Shop']

pi = np.array([0.6,0.4])
A = np.array([[0.7,0.3],[0.4,0.6]])
B = np.array([[0.1, 0.9], [0.6,0.4]])

st.subheader("Initial Parameters")
st.write("**Initial Probabilities ($\pi$)**")
st.dataframe(pd.DataFrame([pi],columns = states))

st.write("**Transition Matrix ($A$)**")
st.dataframe(pd.DataFrame(A,columns = states, index = states))

st.write("**Emission Matrix ($B$)**")
st.dataframe(pd.DataFrame(B, columns = observations, index = states))


#Forward Pass

st.subheader("1. Forward Probability (Prefix Evidence)")

obs_map = {'Walk':0,'Shop':1}
short_sequence = ['Walk','Shop']
obs_indices = [obs_map[obs] for obs in short_sequence]

def forward_pass(obs_seq, pi,A,B):
    T = len(obs_seq)
    N = len(pi)
    alpha = np.zeros((T,N))

    alpha[0, :] = pi * B[:, obs_seq[0]]

    for t in range(1,T):
        for j in range(N):
            alpha[t,j] = np.sum(alpha[t-1, :] * A[:,j]) * B[j, obs_seq[t]]
    return alpha

alpha_matrix = forward_pass(obs_indices,pi,A,B)
st.write(f"**Observation Sequence:** {short_sequence}")

alpha_df = pd.DataFrame(
    alpha_matrix, 
    columns=states, 
    index=[f"t={t+1} ({short_sequence[t]})" for t in range(len(short_sequence))]
)
st.dataframe(alpha_df)

#Backward Pass

st.subheader("2. Backward Probability (Suffix Evidence)")

def backward_pass(obs_seq, A, B):
    T = len(obs_seq)
    N = A.shape[0]
    beta = np.zeros((T,N))

    beta[T-1,:] = 1
    for t in range(T-2,-1,-1):
        for i in range(N):
            beta[t,i] = np.sum(A[i,:] * B[:,obs_seq[t+1]] * beta[t+1,:])
    return beta 

beta_matrix = backward_pass(obs_indices,A,B)


beta_df = pd.DataFrame(
    beta_matrix, 
    columns=states, 
    index=[f"t={t+1} ({short_sequence[t]})" for t in range(len(short_sequence))]
)

st.dataframe(beta_df)


#State Probability

def calc_gamma(alpha, beta):
    P_O = np.sum(alpha[-1,:])

    gamma = (alpha*beta) / P_O
    return P_O, gamma

P_O, gamma_matrix = calc_gamma(alpha_matrix, beta_matrix)

st.write(f"**Total Probability of Observation Sequence $P(O)$:** `{P_O:.4f}`")

st.write("**State Responsibilities ($)")
gamma_df = pd.DataFrame(
    gamma_matrix, 
    columns=states, 
    index=[f"t={t+1} ({short_sequence[t]})" for t in range(len(short_sequence))]
)

st.dataframe(gamma_df)


#Transition Responsibilities
st.subheader("4. Transition Responsibilities ($\\xi$)")

def calc_xi(alpha,beta,A,B, obs_seq, P_O):
    T = len(obs_seq)
    N = A.shape[0]

    xi = np.zeros((T-1, N, N))
    
    for t in range(T-1):
        for i in range(N):
            for j in range(N):
                xi[t, i, j] = (alpha[t, i] * A[i, j] * B[j, obs_seq[t+1]] * beta[t+1, j]) / P_O
                
    return xi

xi_matrix = calc_xi(alpha_matrix, beta_matrix, A, B, obs_indices, P_O)

st.write("**Transition Probabilities at $t=1$ (Walk $\\rightarrow$ Shop)**")
xi_df = pd.DataFrame(
    xi_matrix[0], 
    columns=[f"To {s}" for s in states], 
    index=[f"From {s}" for s in states]
)
st.dataframe(xi_df)


st.subheader("5. Parameter Updates (The Learning Step)")

def update_parameters(gamma, xi, obs_seq, states, observations):
    T = len(obs_seq)
    N = len(states)
    M = len(observations)
    
    pi_new = gamma[0, :]
    
    A_new = np.zeros((N, N))
    for i in range(N):
        gamma_sum = np.sum(gamma[:-1, i]) 
        for j in range(N):
            xi_sum = np.sum(xi[:, i, j])
            A_new[i, j] = xi_sum / gamma_sum if gamma_sum > 0 else 0
            
    B_new = np.zeros((N, M))
    for i in range(N):
        gamma_sum_total = np.sum(gamma[:, i]) 
        for k in range(M):
            gamma_sum_obs = np.sum(gamma[np.array(obs_seq) == k, i])
            B_new[i, k] = gamma_sum_obs / gamma_sum_total if gamma_sum_total > 0 else 0
            
    return pi_new, A_new, B_new

# Calculate the new parameters
pi_new, A_new, B_new = update_parameters(gamma_matrix, xi_matrix, obs_indices, states, observations)

# Display them in Streamlit
st.write("**Updated Initial Probabilities ($\\pi^{new}$)**")
st.dataframe(pd.DataFrame([pi_new], columns=states))

st.write("**Updated Transition Matrix ($A^{new}$)**")
st.dataframe(pd.DataFrame(A_new, columns=states, index=states))

st.write("**Updated Emission Matrix ($B^{new}$)**")
st.dataframe(pd.DataFrame(B_new, columns=observations, index=states))



st.header("6. Full Baum-Welch Iteration (Long Sequence)")

long_sequence = ['Walk', 'Shop', 'Shop', 'Walk', 'Shop']
long_indices = [obs_map[obs] for obs in long_sequence]

# Set up starting parameters again
current_pi = pi.copy()
current_A = A.copy()
current_B = B.copy()

iterations = 5

for i in range(iterations):
    st.write(f"### Iteration {i + 1}")
    
    # 1. Expectation Step
    alpha = forward_pass(long_indices, current_pi, current_A, current_B)
    beta = backward_pass(long_indices, current_A, current_B)
    P_O, gamma = calc_gamma(alpha, beta)
    
    # Calculate xi using the CURRENT iteration's A and B matrices
    T = len(long_indices)
    xi = np.zeros((T-1, len(states), len(states)))
    for t in range(T-1):
        for j in range(len(states)):
            for k in range(len(states)):
                xi[t, j, k] = (alpha[t, j] * current_A[j, k] * current_B[k, long_indices[t+1]] * beta[t+1, k]) / P_O
    
    # 2. Maximization Step
    current_pi, current_A, current_B = update_parameters(gamma, xi, long_indices, states, observations)
    
    # Display the updated Emission Matrix for this iteration
    st.write(f"**Probability of Sequence $P(O)$:** `{P_O:.6f}`")
    st.write("**Updated Emission Matrix ($B$)**")
    st.dataframe(pd.DataFrame(current_B, columns=observations, index=states))
    st.divider()