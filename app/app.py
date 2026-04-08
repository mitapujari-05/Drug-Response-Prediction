import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Drug Response Predictor",
    page_icon="🧬",
    layout="wide"
)

# ── Drug info dictionary ──────────────────────────────────────────
DRUG_INFO = {
    'Erlotinib':  {'target': 'EGFR',    'pathway': 'EGFR signaling',
                   'use': 'Lung & head/neck cancer'},
    'Gefitinib':  {'target': 'EGFR',    'pathway': 'EGFR signaling',
                   'use': 'Non-small cell lung cancer'},
    'Lapatinib':  {'target': 'EGFR/HER2','pathway': 'EGFR signaling',
                   'use': 'Breast cancer (HER2+)'},
    'Cisplatin':  {'target': 'DNA',      'pathway': 'DNA damage response',
                   'use': 'Broad spectrum — lung, bladder, ovarian'},
    'Docetaxel':  {'target': 'Tubulin',  'pathway': 'Mitosis',
                   'use': 'Breast, lung, prostate cancer'},
}

@st.cache_data
def load_expression():
    base = os.path.dirname(__file__)
    return pd.read_csv(os.path.join(base, 'cell_line_expression.csv'),
                       index_col=0)

@st.cache_data
def load_cell_lines():
    base = os.path.dirname(__file__)
    names = pd.read_csv(os.path.join(base, 'cell_line_names.csv'))
    all_names = names.iloc[:, 0].astype(str).tolist()
    # Put named lines first, numeric IDs at bottom
    named   = sorted([n for n in all_names if not n.isdigit()])
    numeric = sorted([n for n in all_names if n.isdigit()])
    return named + numeric

@st.cache_resource
def load_drug_models(drug_name):
    base      = os.path.dirname(__file__)
    model_dir = os.path.join(base, 'models')
    drug_safe = drug_name.replace(' ', '_').replace('/', '_')
    clf    = joblib.load(os.path.join(model_dir, f'{drug_safe}_classifier.pkl'))
    reg    = joblib.load(os.path.join(model_dir, f'{drug_safe}_regressor.pkl'))
    scaler = joblib.load(os.path.join(model_dir, f'{drug_safe}_scaler.pkl'))
    pca    = joblib.load(os.path.join(model_dir, f'{drug_safe}_pca.pkl'))
    return clf, reg, scaler, pca

expr_df    = load_expression()
cell_lines = load_cell_lines()

# ── Header ────────────────────────────────────────────────────────
st.title("🧬 Cancer Drug Response Predictor")
st.markdown(
    "Predicts cancer cell line sensitivity to targeted therapies "
    "using gene expression data from **GDSC2**."
)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.header("🔬 Prediction Input")

selected_drug = st.sidebar.selectbox(
    "Select Drug",
    options=list(DRUG_INFO.keys()),
    index=0
)

# Show drug info
info = DRUG_INFO[selected_drug]
st.sidebar.markdown(f"**Target:** {info['target']}")
st.sidebar.markdown(f"**Pathway:** {info['pathway']}")
st.sidebar.markdown(f"**Clinical use:** {info['use']}")
st.sidebar.markdown("---")

default_idx = cell_lines.index('MCF7') if 'MCF7' in cell_lines else 0
selected_cell_line = st.sidebar.selectbox(
    "Select Cell Line",
    options=cell_lines,
    index=default_idx
)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔬 Predict Response",
                                use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────
if predict_btn:
    if selected_cell_line not in expr_df.index:
        st.error(f"Expression data not found for '{selected_cell_line}'.")
    else:
        with st.spinner(f"Running prediction for {selected_cell_line}..."):
            clf, reg, scaler, pca = load_drug_models(selected_drug)

            x_raw    = expr_df.loc[selected_cell_line].values.reshape(1, -1)
            x_scaled = scaler.transform(x_raw)
            x_pca    = pca.transform(x_scaled)

            prob           = clf.predict_proba(x_pca)[0]
            pred_label     = clf.predict(x_pca)[0]
            ic50_pred      = reg.predict(x_pca)[0]
            sensitivity    = "Sensitive" if pred_label == 1 else "Resistant"
            prob_sensitive = prob[1]

        st.subheader(f"📊 Results: {selected_cell_line} + {selected_drug}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            icon = "🟢" if sensitivity == "Sensitive" else "🔴"
            st.metric("Predicted Response", f"{icon} {sensitivity}")
        with col2:
            st.metric("Predicted LN_IC50", f"{ic50_pred:.3f}")
        with col3:
            st.metric("Sensitivity Probability", f"{prob_sensitive*100:.1f}%")
        with col4:
            st.metric("Drug Target", info['target'])

        st.markdown("---")

        col_left, col_right = st.columns(2)

        # Probability bar chart
        with col_left:
            st.markdown("**Response Probability**")
            fig1, ax1 = plt.subplots(figsize=(6, 2.5))
            colors = ['#2ecc71', '#e74c3c']
            bars = ax1.barh(['Sensitive', 'Resistant'],
                            [prob_sensitive, prob[0]],
                            color=colors, edgecolor='white', height=0.5)
            for bar, val in zip(bars, [prob_sensitive, prob[0]]):
                ax1.text(bar.get_width() + 0.01,
                         bar.get_y() + bar.get_height()/2,
                         f'{val*100:.1f}%', va='center',
                         fontsize=12, fontweight='bold')
            ax1.set_xlim(0, 1.2)
            ax1.axvline(0.5, color='gray', linestyle='--',
                        linewidth=1, alpha=0.5)
            ax1.set_xlabel('Probability')
            plt.tight_layout()
            st.pyplot(fig1)
            plt.close()

        # Gene deviation chart
        with col_right:
            st.markdown("**Top Contributing Genes**")
            mean_expr  = expr_df.mean(axis=0)
            cell_expr  = expr_df.loc[selected_cell_line]
            deviation  = (cell_expr - mean_expr).abs().sort_values(
                ascending=False)
            top12      = deviation.head(12)
            gene_vals  = cell_expr[top12.index] - mean_expr[top12.index]
            colors_g   = ['#2ecc71' if v < 0 else '#e74c3c'
                          for v in gene_vals.values]

            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.barh(top12.index[::-1],
                     gene_vals[top12.index[::-1]],
                     color=colors_g[::-1], edgecolor='white')
            ax2.axvline(0, color='black', linewidth=0.8)
            ax2.set_xlabel('Deviation from mean expression')
            ax2.set_title('Gene expression profile', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")

        # Raw values expander
        with st.expander("View raw gene expression (top 20 genes)"):
            top20 = cell_expr.sort_values(ascending=False).head(20)
            st.dataframe(
                top20.reset_index().rename(
                    columns={'index': 'Gene',
                             selected_cell_line: 'Expression'}),
                use_container_width=True)

else:
    # Landing page
    st.markdown("### How to use")
    st.markdown(
        "1. Select a **drug** from the sidebar\n"
        "2. Select a **cancer cell line**\n"
        "3. Click **Predict Response**\n"
        "4. View predicted sensitivity, IC50, and gene profile"
    )
    st.markdown("---")
    st.markdown("### Supported Drugs")
    drug_df = pd.DataFrame([
        {'Drug': d, 'Target': v['target'],
         'Pathway': v['pathway'], 'Clinical Use': v['use']}
        for d, v in DRUG_INFO.items()
    ])
    st.dataframe(drug_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Model Performance (Erlotinib)")
    perf_df = pd.DataFrame({
        'Model':    ['Random Forest', 'XGBoost', 'MLP'],
        'AUC-ROC':  [0.683, 0.692, 0.651],
        'Accuracy': ['62.0%', '65.4%', '59.8%'],
        'Pearson r':[0.532, 0.506, 0.483]
    })
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption(
        "Data: GDSC2 | 891 cell lines | 17,737 genes | "
        "Models: XGBoost (classifier) + Random Forest (regressor)"
    )