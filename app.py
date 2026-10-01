import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from predict import predict_code, predict_metrics, load_model
from code2vec_extractor import extractor

# ============================================================
# PAGE CONFIGURATION & STYLING
# ============================================================

st.set_page_config(
    page_title="Hybrid LinkNet-BiLSTM Software Defect Prediction",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .paper-badge {
        background: linear-gradient(135deg, #1E40AF, #3B82F6);
        color: white;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .status-defective {
        background-color: #FEE2E2;
        border: 1px solid #EF4444;
        color: #991B1B;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
    .status-clean {
        background-color: #DCFCE7;
        border: 1px solid #22C55E;
        color: #166534;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="paper-badge">IJECE Journal 2025</div>', unsafe_allow_html=True)
    st.title("System Overview")
    st.markdown("""
    **Paper Title:**  
    *Hybrid Deep Learning Model for Software Defect Classification Using Code2Vec and LinkNet-BiLSTM Score Level Fusion*
    
    **Authors:**  
    Srinivasa Rao Katragadda & Sirisha Potluri  
    *Koneru Lakshmaiah Education Foundation, India*
    
    ---
    ### ⚙️ Model Architecture
    - **Code2Vec:** AST Path Extraction & Attention Aggregation
    - **Improved LinkNet:** Dilated Convolutions & Skip Connections
    - **Bi-LSTM:** Forward & Backward Sequence Dependencies
    - **Fusion:** Score-Level Weighted Averaging ($w_1=0.5, w_2=0.5$)
    """)

    st.divider()
    model_status = "✅ Model Ready" if os.path.exists(os.path.join("models", "defect_model.pth")) else "❌ Model Missing"
    st.markdown(f"**Status:** {model_status}")


# ============================================================
# HEADER
# ============================================================

st.markdown('<div class="main-title">Software Defect Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hybrid Code2Vec + Improved LinkNet + Bi-LSTM with Score-Level Fusion</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💻 Raw Source Code Analysis (Code2Vec)",
    "📊 PROMISE Software Metrics (CM1)",
    "📈 Paper Benchmark & Results",
    "📐 Architecture & Equations"
])


# ============================================================
# TAB 1: CODE2VEC RAW SOURCE CODE PREDICTION
# ============================================================

with tab1:
    st.header("Source Code AST & Defect Analysis")
    st.write("Extract syntactic and semantic features from raw code snippets using **Code2Vec** and classify defect propensity via **LinkNet + Bi-LSTM**.")

    sample_codes = {
        "Clean Helper Function (No Defect)": """def calculate_discount(price, rate):
    if rate < 0 or rate > 1:
        raise ValueError("Invalid discount rate")
    return price * (1.0 - rate)""",

        "Recursive Function with Stack Overflow Risk (Defective)": """def recursive_deep_parser(data, depth):
    if depth > 50000:
        return data
    # Missing base termination case on null data
    return recursive_deep_parser(data.next, depth + 1)""",

        "Division by Zero / Unhandled Exception (Defective)": """def compute_batch_average(values):
    total = sum(values)
    count = len(values)
    # Bug: Unchecked division when list is empty
    return total / count""",

        "Resource Leak / Unclosed Handle (Defective)": """def write_log(filename, message):
    file = open(filename, 'a')
    file.write(message)
    # Bug: Missing file.close() or with statement context"""
    }

    selected_sample = st.selectbox("Select a Preset Code Snippet or Enter Custom Code:", list(sample_codes.keys()))
    code_input = st.text_area("Source Code Snippet:", value=sample_codes[selected_sample], height=180)

    if st.button("🚀 Analyze Source Code with Code2Vec & Hybrid Model", type="primary", use_container_width=True):
        if not code_input.strip():
            st.warning("Please enter some source code.")
        else:
            with st.spinner("Extracting AST Paths and Running Score-Level Fusion..."):
                res = predict_code(code_input)

            st.divider()
            col_pred1, col_pred2 = st.columns([1, 1])

            with col_pred1:
                if res["prediction_label"] == "Defective":
                    st.markdown(f'<div class="status-defective">⚠️ Predicted: DEFECTIVE<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {res["confidence"]*100:.2f}%</span></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-clean">✅ Predicted: NON-DEFECTIVE<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {res["confidence"]*100:.2f}%</span></div>', unsafe_allow_html=True)

            with col_pred2:
                # Score-level fusion breakdown
                st.subheader("🎯 Score-Level Fusion Breakdown")
                fusion_df = pd.DataFrame({
                    "Model Component": ["Improved LinkNet", "Bi-LSTM", "Fused Score (w1=0.5, w2=0.5)"],
                    "Non-Defective Prob": [f"{res['linknet_non_defective']*100:.1f}%", f"{res['bilstm_non_defective']*100:.1f}%", f"{res['non_defective_probability']*100:.1f}%"],
                    "Defective Prob": [f"{res['linknet_defective']*100:.1f}%", f"{res['bilstm_defective']*100:.1f}%", f"{res['defective_probability']*100:.1f}%"]
                })
                st.dataframe(fusion_df, hide_index=True, use_container_width=True)

            # AST Path details
            with st.expander(f"🌿 Code2Vec AST Path Breakdown ({res['ast_contexts_count']} Context Paths Extracted)", expanded=True):
                st.write("Extracted AST Triples `(Token 1, AST Path, Token 2)`:")
                for idx, (t1, path, t2) in enumerate(res["sample_ast_paths"]):
                    st.markdown(f"**Path {idx+1}:** `{t1}` ➔ `{path}` ➔ `{t2}`")


# ============================================================
# TAB 2: PROMISE TABULAR METRICS PREDICTOR (CM1)
# ============================================================

with tab2:
    st.header("PROMISE Repository Software Metrics (CM1)")
    st.write("Predict defect probability from 21 McCabe Complexity and Halstead static code metrics.")

    features_path = os.path.join("models", "features.pkl")
    if os.path.exists(features_path):
        feature_cols = joblib.load(features_path)
    else:
        feature_cols = ["loc", "v(g)", "ev(g)", "iv(g)", "n", "v", "l", "d", "i", "e", "b", "t", "lOCode", "lOComment", "lOBlank", "locCodeAndComment", "uniq_Op", "uniq_Opnd", "total_Op", "total_Opnd", "branchCount"]

    # Sample presets
    col_pre1, col_pre2 = st.columns([1, 1])
    preset_defective = {
        "loc": 46.0, "v(g)": 15.0, "ev(g)": 3.0, "iv(g)": 1.0, "n": 239.0, "v": 1362.41,
        "l": 0.04, "d": 22.3, "i": 61.1, "e": 30377.95, "b": 0.45, "t": 1687.66,
        "lOCode": 8.0, "lOComment": 35.0, "lOBlank": 22.0, "locCodeAndComment": 0.0,
        "uniq_Op": 15.0, "uniq_Opnd": 37.0, "total_Op": 129.0, "total_Opnd": 110.0, "branchCount": 29.0
    }
    preset_clean = {
        "loc": 7.0, "v(g)": 1.0, "ev(g)": 1.0, "iv(g)": 1.0, "n": 11.0, "v": 34.87,
        "l": 0.5, "d": 2.0, "i": 17.43, "e": 69.74, "b": 0.01, "t": 3.87,
        "lOCode": 0.0, "lOComment": 0.0, "lOBlank": 1.0, "locCodeAndComment": 0.0,
        "uniq_Op": 4.0, "uniq_Opnd": 5.0, "total_Op": 6.0, "total_Opnd": 5.0, "branchCount": 1.0
    }

    preset_choice = st.radio("Load Metric Template:", ["Custom Input", "Defective Module Sample (High Complexity)", "Clean Module Sample (Low Complexity)"], horizontal=True)

    input_vals = {}
    cols = st.columns(3)
    for idx, feature in enumerate(feature_cols):
        col = cols[idx % 3]
        default_val = 0.0
        if preset_choice == "Defective Module Sample (High Complexity)":
            default_val = preset_defective.get(feature, 0.0)
        elif preset_choice == "Clean Module Sample (Low Complexity)":
            default_val = preset_clean.get(feature, 0.0)

        with col:
            input_vals[feature] = st.number_input(f"{feature}:", value=float(default_val), format="%.2f", key=f"tab2_{feature}")

    if st.button("📊 Evaluate Software Metrics", type="primary", use_container_width=True):
        with st.spinner("Standardizing features & calculating LinkNet-BiLSTM fusion..."):
            res = predict_metrics(input_vals)

        st.divider()
        c1, c2 = st.columns([1, 1])
        with c1:
            if res["prediction_label"] == "Defective":
                st.markdown(f'<div class="status-defective">⚠️ Predicted: DEFECTIVE<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {res["confidence"]*100:.2f}%</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-clean">✅ Predicted: NON-DEFECTIVE<br><span style="font-size:0.9rem; font-weight:normal;">Confidence: {res["confidence"]*100:.2f}%</span></div>', unsafe_allow_html=True)

        with c2:
            st.subheader("Model Probability Breakdown")
            chart_df = pd.DataFrame({
                "Class": ["Non-Defective", "Defective"],
                "LinkNet Prob": [res["linknet_non_defective"], res["linknet_defective"]],
                "Bi-LSTM Prob": [res["bilstm_non_defective"], res["bilstm_defective"]],
                "Fused Prob": [res["non_defective_probability"], res["defective_probability"]]
            })
            st.bar_chart(chart_df.set_index("Class"))


# ============================================================
# TAB 3: BENCHMARK & EXPERIMENTAL RESULTS (MATCHING PAPER)
# ============================================================

with tab3:
    st.header("Experimental Results & Comparisons (Paper Section 5)")

    # Table 1 replication
    st.subheader("📋 Table 1: Performance Comparison of Methods")
    comparison_data = {
        "Method": ["CNN + MLP", "RNN", "Rule-Based Algorithm", "Proposed (Hybrid LinkNet-BiLSTM)"],
        "Accuracy (%)": [85, 78, 81, 92],
        "Precision (%)": [84, 76, 79, 90],
        "Recall (%)": [77, 89, 73, 89],
        "F1 Score (%)": [83, 78, 81, 90]
    }
    st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)

    col_fig1, col_fig2 = st.columns(2)

    with col_fig1:
        st.subheader("⏱️ Training Time Comparison (Figure 7)")
        time_df = pd.DataFrame({
            "Model": ["Improved LinkNet", "Bi-LSTM", "Hybrid (Proposed)"],
            "Training Time (seconds)": [120, 150, 200]
        })
        st.bar_chart(time_df.set_index("Model"))

    with col_fig2:
        st.subheader("🎯 Tuning Improvement (Figure 6)")
        tuning_df = pd.DataFrame({
            "Stage": ["Before Tuning", "After Tuning"],
            "Accuracy": [0.85, 0.92]
        })
        st.bar_chart(tuning_df.set_index("Stage"))

    # Active Training History
    history_path = os.path.join("models", "training_history.json")
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            hist = json.load(f)

        st.divider()
        st.subheader("📉 Actual Training & Validation Convergence Curves (Figures 4 & 5)")
        c_curve1, c_curve2 = st.columns(2)
        with c_curve1:
            st.line_chart(pd.DataFrame({
                "Train Loss": hist["train_loss"],
                "Validation Loss": hist["test_loss"]
            }))
        with c_curve2:
            st.line_chart(pd.DataFrame({
                "Train Accuracy": hist["train_accuracy"],
                "Validation Accuracy": hist["test_accuracy"]
            }))


# ============================================================
# TAB 4: MATHEMATICAL FORMULATIONS
# ============================================================

with tab4:
    st.header("Mathematical Formulations from Research Paper")

    st.subheader("1. Improved LinkNet Dilated Convolutions (Equations 1 & 2)")
    st.latex(r"ks_{eff} = ks + (ks + 1) \times (dr - 1) = dr \times (ks - 1) + 1 \quad \text{(Eq 1)}")
    st.latex(r"rf_n = x_{n-1} \times rf_{n-1} + ks_{n-1} - x_{n-1} \quad \text{(Eq 2)}")
    st.caption("Dilated convolutions expand the receptive field without increasing the parameter count, preserving structural spatial relationships.")

    st.divider()
    st.subheader("2. Bi-LSTM Cell Computations (Equations 3 to 10)")
    st.latex(r"f_t = \sigma(W_{fh} [h_{t-1}] + M_{fa} [a_t] + b_f) \quad \text{(Forget Gate, Eq 3)}")
    st.latex(r"i_t = \sigma(w_{ih} [h_{t-1}] + u_{ix} [x_t] + b_i) \quad \text{(Input Gate, Eq 4)}")
    st.latex(r"c_t = f_t * c_{t-1} + i_t * \tanh(w_{ch} [h_{t-1}] + u_{cx} [x_t]) \quad \text{(Cell State, Eq 6)}")
    st.latex(r"h_t = [\vec{h_t}; \overleftarrow{h_t}] \quad \text{(Bidirectional Hidden State, Eq 9)}")
    st.latex(r"y = \text{softmax}(W_y h_t + b_y) \quad \text{(Class Probability, Eq 10)}")

    st.divider()
    st.subheader("3. Score-Level Fusion (Equation 11)")
    st.latex(r"S_{final} = w_1 \cdot S_{linknet} + w_2 \cdot S_{Bi-LSTM} \quad \text{where } w_1 + w_2 = 1")
    st.caption("Combines structural pattern identification with sequential temporal dependencies to reduce noise and enhance classification accuracy.")