import streamlit as st
import numpy as np
from hmm import HMM
from visualization import get_convergence_plot, get_transition_diagram, get_emissions_plot

def main():
    st.set_page_config(page_title="HMM Baum-Welch UI", layout="wide")
    st.title("Hidden Markov Model (Baum-Welch)")

    with st.sidebar:
        st.header("Input Parameters")
        obs_seq_input = st.text_input("Observation Sequence (comma-separated)", "0, 1, 0, 1, 1, 0")
        n_states = st.number_input("Number of Hidden States", min_value=1, value=2)
        n_iter = st.number_input("Iterations", min_value=1, value=50)
        train_btn = st.button("Train HMM", type="primary", use_container_width=True)

    if train_btn:
        try:
            obs_seq = np.array([int(x.strip()) for x in obs_seq_input.split(",")])
            n_obs_types = int(np.max(obs_seq)) + 1
            if n_obs_types < 2: n_obs_types = 2 

            hmm = HMM(n_states=int(n_states), n_observations=n_obs_types)
            
            with st.spinner("Training model using Baum-Welch algorithm..."):
                log_likelihoods = hmm.baum_welch(obs_seq, n_iter=int(n_iter))
                params = hmm.get_params()
                alpha, scaling_factors = hmm._forward(obs_seq)
                log_prob_O = -np.sum(np.log(scaling_factors))

            st.success("Training Complete!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Log-Likelihood", f"{log_likelihoods[0]:.4f}")
            col2.metric("Final Log-Likelihood", f"{log_likelihoods[-1]:.4f}")
            col3.metric("Final P(O | λ)", f"{np.exp(log_prob_O):.4e}")

            st.subheader("Learned Parameters")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.write("**Transition Matrix (A)**")
                st.dataframe(params["A"])
            with col_b:
                st.write("**Emission Matrix (B)**")
                st.dataframe(params["B"])
            with col_c:
                st.write("**Initial Distribution (π)**")
                st.dataframe(params["pi"])

            st.subheader("Visualizations")
            
            with st.spinner("Generating plots..."):
                fig_convergence = get_convergence_plot(log_likelihoods)
                dot_transitions = get_transition_diagram(params["A"])
                fig_emissions = get_emissions_plot(params["B"])

            tab1, tab2, tab3 = st.tabs(["Convergence Plot", "State Transitions", "Emission Matrix"])
            
            with tab1:
                st.pyplot(fig_convergence)
            with tab2:
                try:
                    st.graphviz_chart(dot_transitions)
                except Exception as e:
                    st.info(f"Could not render Graphviz diagram: {e}. Please ensure Graphviz is installed.")
            with tab3:
                st.pyplot(fig_emissions)

        except Exception as e:
            st.error(f"Error during training: {e}")

if __name__ == "__main__":
    main()
