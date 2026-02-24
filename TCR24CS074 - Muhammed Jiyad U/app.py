import streamlit as st
import numpy as np
from hmmlearn import hmm
import graphviz

st.title("HMM Visualizer: Baum-Welch Algorithm")
st.markdown("This app trains a Hidden Markov Model using the Baum-Welch algorithm and visualizes the state transitions.")

st.sidebar.header("Model Parameters")
n_states = st.sidebar.number_input("Number of Hidden States", min_value=2, max_value=5, value=2)
n_iter = st.sidebar.number_input("Baum-Welch Iterations", min_value=10, max_value=1000, value=100)

st.sidebar.markdown("---")
st.sidebar.markdown("**Observation Sequence**")
obs_input = st.sidebar.text_input("Sequence (comma-separated)", "0,1,0,1,1,0,0,1,0,1")

if st.button("Train Model & Generate Diagram"):
    try:
        obs_list = [int(x.strip()) for x in obs_input.split(',')]
        obs_array = np.array([obs_list]).T 
        
        model = hmm.CategoricalHMM(n_components=n_states, n_iter=n_iter)
        model.fit(obs_array)
        
        st.success(f"Model trained successfully!")
        st.subheader("Transition Probability Matrix")
        st.dataframe(model.transmat_)
        
        st.subheader("State Transition Diagram")
        dot = graphviz.Digraph()
        dot.attr(rankdir='LR')
        
        for i in range(n_states):
            dot.node(str(i), f"State {i}\nPrior: {model.startprob_[i]:.2f}", shape='circle')
            
        for i in range(n_states):
            for j in range(n_states):
                prob = model.transmat_[i, j]
                if prob > 0.01: 
                    dot.edge(str(i), str(j), label=f"{prob:.2f}")
                    
        st.graphviz_chart(dot)

    except Exception as e:
        st.error("Error: Please ensure your sequence contains only integers.")
      
