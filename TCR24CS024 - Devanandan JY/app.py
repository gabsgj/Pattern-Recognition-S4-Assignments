import streamlit as st
import numpy as np
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import networkx as nx
import time

# --- Initialize Session State ---
if 'history' not in st.session_state:
    st.session_state.history = None
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'playing' not in st.session_state:
    st.session_state.playing = False
if 'speed' not in st.session_state:
    st.session_state.speed = 2.0

# --- Core HMM Class ---
class HMM_BaumWelch:
    def __init__(self, n_hidden_states, n_observations):
        self.N = n_hidden_states
        self.M = n_observations
        self.pi = np.random.dirichlet(np.ones(self.N))
        self.A = np.random.dirichlet(np.ones(self.N), size=self.N)
        self.B = np.random.dirichlet(np.ones(self.M), size=self.N)

    def forward(self, obs):
        T = len(obs)
        alpha = np.zeros((T, self.N))
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.B[i, obs[0]]
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, obs[t]]
        return alpha

    def backward(self, obs):
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[T-1] = 1.0
        for t in range(T-2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs[t+1]] * beta[t+1, :])
        return beta

    def train(self, obs, iterations=50):
        T = len(obs)
        history = []
        
        for it in range(iterations + 1):
            alpha = self.forward(obs)
            beta = self.backward(obs)
            p_obs = np.sum(alpha[T-1])
            
            history.append({
                'iteration': it,
                'pi': self.pi.copy(),
                'A': self.A.copy(),
                'B': self.B.copy(),
                'p_obs': p_obs
            })
            
            if it == iterations or p_obs == 0: 
                break

            gamma = np.zeros((T, self.N))
            xi = np.zeros((T-1, self.N, self.N))

            for t in range(T):
                gamma[t] = (alpha[t] * beta[t]) / p_obs
            for t in range(T-1):
                for i in range(self.N):
                    for j in range(self.N):
                        xi[t, i, j] = (alpha[t, i] * self.A[i, j] * self.B[j, obs[t+1]] * beta[t+1, j]) / p_obs

            self.pi = gamma[0]
            for i in range(self.N):
                gamma_sum = np.sum(gamma[:-1, i])
                if gamma_sum > 0:
                    self.A[i, :] = np.sum(xi[:, i, :], axis=0) / gamma_sum
            for j in range(self.N):
                gamma_sum_all = np.sum(gamma[:, j])
                if gamma_sum_all > 0:
                    for k in range(self.M):
                        self.B[j, k] = np.sum(gamma[np.array(obs) == k, j]) / gamma_sum_all
            
        return history

# --- Streamlit UI Build ---
st.set_page_config(page_title="HMM Visualizer", layout="wide")

# --- CSS Injection for Sticky Media Player & Stylish Buttons ---
st.markdown("""
<style>
    div.block-container { padding-bottom: 120px; }
    div[data-testid="stVerticalBlock"]:has(#media-player-anchor) {
        position: sticky; bottom: 0; background: rgba(14, 17, 23, 0.85); backdrop-filter: blur(12px); 
        z-index: 999; padding: 15px 20px 25px 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px 20px 0 0; box-shadow: 0px -10px 40px rgba(0, 0, 0, 0.8);
    }
    div[data-testid="stVerticalBlock"]:has(#media-player-anchor) div.stButton > button[kind="secondary"] {
        border-radius: 30px; border: 1px solid rgba(255, 255, 255, 0.15); background-color: rgba(255, 255, 255, 0.05);
        color: white; font-weight: 600; height: 45px; transition: all 0.25s ease;
    }
    div[data-testid="stVerticalBlock"]:has(#media-player-anchor) div.stButton > button[kind="secondary"]:hover {
        border-color: #00E5FF; color: #00E5FF; background-color: rgba(0, 229, 255, 0.08);
        box-shadow: 0px 4px 15px rgba(0, 229, 255, 0.3); transform: translateY(-2px); 
    }
    div[data-testid="stVerticalBlock"]:has(#media-player-anchor) div.stButton > button[kind="primary"] {
        border-radius: 30px; background: linear-gradient(135deg, #2979FF, #00E5FF); border: none;
        color: #050505; font-weight: 800; height: 45px; box-shadow: 0px 4px 15px rgba(0, 229, 255, 0.3); transition: all 0.25s ease;
    }
    div[data-testid="stVerticalBlock"]:has(#media-player-anchor) div.stButton > button[kind="primary"]:hover {
        box-shadow: 0px 6px 20px rgba(0, 229, 255, 0.6); transform: scale(1.05); 
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Inputs & Legend ---
with st.sidebar:
    st.title("HMM Parameters")
    n_states = st.number_input("Hidden States (N)", min_value=2, max_value=10, value=3)
    n_obs_symbols = st.number_input("Observation Symbols (M)", min_value=2, max_value=10, value=3)
    iterations = st.slider("Max Iterations", 10, 500, 50)
    seq_input = st.text_input("Sequence", value="0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2")
    
    if st.button("Generate Timeline", type="primary"):
        try:
            obs_seq = [int(x.strip()) for x in seq_input.split(',')]
            if max(obs_seq) >= n_obs_symbols:
                st.error(f"Error: max symbol allowed is {n_obs_symbols-1}.")
            else:
                with st.spinner("Calculating..."):
                    np.random.seed(42) 
                    hmm = HMM_BaumWelch(n_hidden_states=n_states, n_observations=n_obs_symbols)
                    st.session_state.history = hmm.train(obs_seq, iterations=iterations)
                    st.session_state.step = 0
                    st.session_state.playing = False
        except ValueError:
            st.error("Invalid input.")

    st.divider()
    st.markdown("### 🗺️ Graph Legend")
    st.markdown("""
    **Shapes:**
    * 🟦 **Blue Squircle:** Hidden State
    * 🟡 **Yellow Circle:** Observation Symbol
    * ⬛ **Dark Box:** START Node
    
    **Wires (Probabilities):**
    * 🟩 **Green Arrow:** State Transition ($A$)
    * 🟥 **Red Arrow:** Symbol Emission ($B$)
    * ⬜ **Dashed Arrow:** Initial State ($\pi$)
    * 🩵 **Cyan Glow:** Value updated this step!
    """)

# --- Main Dashboard ---
if st.session_state.history is not None:
    history = st.session_state.history
    
    hist_n_states = len(history[0]['pi'])
    hist_n_obs = history[0]['B'].shape[1]
    
    if hist_n_states != n_states or hist_n_obs != n_obs_symbols:
        st.info("🔄 Model parameters changed! Please click **Generate Timeline** to run the updated model.")
    else:
        max_steps = len(history) - 1
        T_len = len([int(x.strip()) for x in seq_input.split(',')])

        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            current = history[st.session_state.step]
            prev = history[max(0, st.session_state.step - 1)]
            state_labels = [f"hs{i+1}" for i in range(n_states)]
            obs_labels = [f"em{i+1}" for i in range(n_obs_symbols)]
            
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("**Initial ($\pi$)**")
                st.dataframe(pd.DataFrame(current['pi'], index=state_labels, columns=["Prob"]), height=140, use_container_width=True)
            with m2:
                st.markdown("**Transition ($A$)**")
                st.dataframe(pd.DataFrame(current['A'], index=state_labels, columns=state_labels), height=140, use_container_width=True)
            with m3:
                st.markdown("**Emission ($B$)**")
                st.dataframe(pd.DataFrame(current['B'], index=state_labels, columns=obs_labels), height=140, use_container_width=True)

            p_all = np.array([max(h['p_obs'], 1e-300) for h in history])
            steps = list(range(st.session_state.step + 1))
            p_curr = p_all[:st.session_state.step + 1]
            
            log_p = np.log(p_curr)
            nll = -log_p
            error_rate = 1 - np.power(p_curr, 1.0 / T_len)

            fig = make_subplots(rows=2, cols=2, subplot_titles=('Log-Likelihood', 'Probability P(O|λ)', 'Loss (Negative Log)', 'Error Rate'), vertical_spacing=0.15, horizontal_spacing=0.1)
            
            fig.add_trace(go.Scatter(x=steps, y=log_p, mode='lines+markers', line=dict(color='#2979FF', width=2), name="Log-Likelihood"), row=1, col=1)
            fig.add_trace(go.Scatter(x=steps, y=p_curr, mode='lines+markers', line=dict(color='#00E676', width=2), name="Probability"), row=1, col=2)
            fig.add_trace(go.Scatter(x=steps, y=nll, mode='lines+markers', line=dict(color='#FF5252', width=2), name="Loss"), row=2, col=1)
            fig.add_trace(go.Scatter(x=steps, y=error_rate, mode='lines+markers', line=dict(color='#E040FB', width=2), name="Error Rate"), row=2, col=2)

            fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, height=450, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified")
            fig.update_xaxes(range=[0, max_steps], showgrid=True, gridwidth=1, gridcolor='#333333')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333333')

            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # --- THE HYBRID ENGINE LOGIC ---
            if st.session_state.playing:
                # 1. PLAYBACK MODE (Matplotlib - Zero Flicker)
                st.markdown("**Live State Diagram** *(Playback Mode - Flicker-Free)*")
                
                fig_net, ax_net = plt.subplots(figsize=(7, 6.5))
                fig_net.patch.set_facecolor('#0e1117')
                ax_net.set_facecolor('#0e1117')
                
                G = nx.DiGraph()
                pos = {}
                
                # Math Coordinates
                pos["START"] = (0, 2)
                for i in range(n_states): pos[state_labels[i]] = ((i - (n_states - 1) / 2) * 1.5, 1)
                for k in range(n_obs_symbols): pos[obs_labels[k]] = ((k - (n_obs_symbols - 1) / 2) * 1.5, 0)
                
                G.add_nodes_from(["START"] + state_labels + obs_labels)
                
                # Draw Nodes to match Pyvis exactly
                nx.draw_networkx_nodes(G, pos, nodelist=["START"], node_color="#555555", node_shape="s", node_size=1200, ax=ax_net)
                nx.draw_networkx_nodes(G, pos, nodelist=state_labels, node_color="#1E88E5", node_shape="s", node_size=2000, edgecolors="white", ax=ax_net)
                nx.draw_networkx_nodes(G, pos, nodelist=obs_labels, node_color="#FFCA28", node_shape="o", node_size=1500, ax=ax_net)
                nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", font_size=10, ax=ax_net)
                
                edges, colors, widths = [], [], []
                edge_labels = {}
                
                def add_e(u, v, prob, prev_prob, active_color, inactive_color):
                    is_upd = abs(prob - prev_prob) > 0.005 and st.session_state.step > 0
                    c = "#00E5FF" if is_upd else (active_color if prob > 0.01 else inactive_color)
                    w = (prob * 4) + (2.5 if is_upd else 1.5)
                    edges.append((u, v))
                    colors.append(c)
                    widths.append(w)
                    edge_labels[(u, v)] = f"{prob:.2f}"
                    
                for i in range(n_states): add_e("START", state_labels[i], current['pi'][i], prev['pi'][i], "#AAAAAA", "#555555")
                for i in range(n_states):
                    for j in range(n_states): add_e(state_labels[i], state_labels[j], current['A'][i, j], prev['A'][i, j], "#00E676", "#3b6b4c")
                for i in range(n_states):
                    for k in range(n_obs_symbols): add_e(state_labels[i], obs_labels[k], current['B'][i, k], prev['B'][i, k], "#FF5252", "#803535")
                    
                nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=colors, width=widths, arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.15", ax=ax_net)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="white", font_size=8, label_pos=0.3, bbox=dict(facecolor="#0e1117", edgecolor="none", pad=0), ax=ax_net)
                
                ax_net.axis('off')
                fig_net.tight_layout()
                st.pyplot(fig_net)

            else:
                # 2. PAUSED MODE (Pyvis - Interactive & Draggable)
                st.markdown("**Live State Diagram** *(Paused - Drag nodes to explore)*")
                
                net = Network(height='550px', width='100%', directed=True, bgcolor='#0e1117', font_color='white')
                net.add_node("START", label="START", shape="box", color="#555555", x=0, y=-240)
                
                for i in range(n_states):
                    x_pos = (i - (n_states - 1) / 2) * 260
                    net.add_node(state_labels[i], label=state_labels[i], shape="box", color="#1E88E5", x=x_pos, y=0)
                    prob_pi = current['pi'][i]
                    is_upd = abs(prob_pi - prev['pi'][i]) > 0.005 and st.session_state.step > 0
                    color_pi = "#00E5FF" if is_upd else ("#AAAAAA" if prob_pi > 0.01 else "#555555")
                    net.add_edge("START", state_labels[i], label=f"{prob_pi:.2f}", color=color_pi, dashes=True, width=(prob_pi*4)+1.5)

                for k in range(n_obs_symbols):
                    x_pos = (k - (n_obs_symbols - 1) / 2) * 260
                    net.add_node(obs_labels[k], label=obs_labels[k], shape="circle", color="#FFCA28", size=45, x=x_pos, y=240)

                for i in range(n_states):
                    for j in range(n_states):
                        prob_A = current['A'][i, j]
                        is_upd = abs(prob_A - prev['A'][i, j]) > 0.005 and st.session_state.step > 0
                        color_A = "#00E5FF" if is_upd else ("#00E676" if prob_A > 0.01 else "#3b6b4c")
                        net.add_edge(state_labels[i], state_labels[j], label=f"{prob_A:.2f}", color=color_A, width=(prob_A*4)+(2.5 if is_upd else 1.5))
                            
                for i in range(n_states):
                    for k in range(n_obs_symbols):
                        prob_B = current['B'][i, k]
                        is_upd = abs(prob_B - prev['B'][i, k]) > 0.005 and st.session_state.step > 0
                        color_B = "#00E5FF" if is_upd else ("#FF5252" if prob_B > 0.01 else "#803535")
                        net.add_edge(state_labels[i], obs_labels[k], label=f"{prob_B:.2f}", color=color_B, width=(prob_B*4)+(2.5 if is_upd else 1.5))

                net.set_options("""
                {
                  "physics": {"enabled": false},
                  "interaction": {"dragNodes": true},
                  "nodes": {"margin": 15, "font": {"size": 22, "color": "white"}},
                  "edges": {"smooth": {"type": "curvedCW", "roundness": 0.2}, "font": {"size": 15, "align": "middle", "color": "white", "background": "#0e1117", "strokeWidth": 0}}
                }
                """)
                
                try:
                    net.save_graph("pyvis_graph.html")
                    with open("pyvis_graph.html", 'r', encoding='utf-8') as f: html_data = f.read()
                    html_data = html_data.replace('<style type="text/css">', '<style type="text/css">\nbody { background-color: #0e1117 !important; }\n')
                    components.html(html_data, height=570)
                except Exception as e:
                    st.error(f"Graph rendering error: {e}")

        # --- Stylish Fixed Media Player (Bottom Center) ---
        player_container = st.container()
        with player_container:
            st.markdown("<span id='media-player-anchor'></span>", unsafe_allow_html=True)
            
            c_iter, c_back, c_play, c_next, c_speed, c_slider = st.columns([1.2, 1, 1, 1, 1, 6], vertical_alignment="center")
            
            with c_iter: st.markdown(f"<div style='text-align: center; font-size: 1.1rem; font-weight: bold;'>Step {st.session_state.step}/{max_steps}</div>", unsafe_allow_html=True)
            with c_back: 
                if st.button("⏮️ Back", use_container_width=True): st.session_state.step = max(0, st.session_state.step - 1); st.session_state.playing = False; st.rerun()
            with c_play:
                if st.button("⏸️ Pause" if st.session_state.playing else "▶️ Play", type="primary", use_container_width=True): st.session_state.playing = not st.session_state.playing; st.rerun()
            with c_next:
                if st.button("⏭️ Next", use_container_width=True): st.session_state.step = min(max_steps, st.session_state.step + 1); st.session_state.playing = False; st.rerun()
            with c_speed:
                speeds = [1.0, 2.0, 5.0, 10.0]
                if st.button(f"⏩ {st.session_state.speed}x", use_container_width=True): 
                    idx = speeds.index(st.session_state.speed)
                    st.session_state.speed = speeds[(idx + 1) % len(speeds)]
                    st.rerun()
            with c_slider:
                scrub = st.slider("Timeline", 0, max_steps, st.session_state.step, label_visibility="collapsed")
                if scrub != st.session_state.step: st.session_state.step = scrub; st.session_state.playing = False; st.rerun()

# --- Animation Loop ---
if st.session_state.playing:
    if st.session_state.step < max_steps:
        time.sleep(max(0.01, 0.5 / st.session_state.speed))
        st.session_state.step += 1
        st.rerun()
    else:
        st.session_state.playing = False
        st.rerun()
