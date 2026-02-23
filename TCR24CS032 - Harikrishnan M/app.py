import streamlit as st
import numpy as np
import pandas as pd
from baum_welch import HMM
# Import your friend's library
from state_transition_diagrams import render_state_diagram, DiagramConfig

st.set_page_config(page_title="Advanced HMM Visualizer", layout="wide")
st.title("📊 HMM Baum-Welch + Interactive Diagrams")

obs_input = st.text_input("Enter Observation Sequence (0, 1...)", "0,1,0,0,1,1,0")
obs_seq = np.array([int(x.strip()) for x in obs_input.split(",")])

if st.button("🚀 Train & Render Diagram"):
    # 1. Run your Baum-Welch Math
    A_init = [[0.6, 0.4], [0.3, 0.7]]
    B_init = [[0.2, 0.8], [0.7, 0.3]]
    pi_init = [0.5, 0.5]
    
    model = HMM(A_init, B_init, pi_init)
    A_final, B_final, pi_final = model.train(obs_seq)
    
    # 2. Use Gabriel's Library to create the diagram
    st.subheader("State Transition Diagram (Professional Render)")
    
    cfg = DiagramConfig(
        state_colors=["#E74C3C", "#3498DB"],
        title="Trained HMM Model",
        background_color="#FAFAFA",
        layout_engine="circo" # This makes it look like a neat circle
    )
    
    # Generate the Graphviz object using the library
    dot = render_state_diagram(A_final, state_labels=["State A", "State B"], config=cfg)
    
    # Show it in Streamlit
    st.graphviz_chart(dot.source)
    
    # 3. Show the raw matrices
    col1, col2 = st.columns(2)
    with col1:
        st.write("Transition Matrix (A)", pd.DataFrame(A_final))
    with col2:
        st.write("Emission Matrix (B)", pd.DataFrame(B_final))

    st.success("Model trained and diagram rendered using state-transition-diagrams library!")