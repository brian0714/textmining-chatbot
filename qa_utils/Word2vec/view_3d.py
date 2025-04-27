import time
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import streamlit as st
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt

def init_session_state(sentences):
    st.session_state.setdefault("selected_indices_3d", [0, 1])
    st.session_state.setdefault("trigger_plot_3d", False)
    st.session_state.setdefault("sentence_picker", [f"Sentence {i+1}: {s}" for i, s in enumerate(sentences[:2])])

def multiselect_changed(key, new_value):
    if st.session_state.get(key) != new_value:
        st.session_state[key] = new_value
        st.session_state["trigger_plot_3d"] = False

def _draw_scatter(reduced_vectors, model, word_colors):
    return go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers+text',
        text=model.wv.index_to_key,
        marker=dict(color=word_colors, size=3),
        hovertemplate="Word: %{text}",
        name="Words"
    )

def _draw_lines(reduced_vectors, model, tokenized_sentences, hex_colors):
    traces = []
    for i in st.session_state["selected_indices_3d"]:
        if i >= len(tokenized_sentences):
            continue
        line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in tokenized_sentences[i] if word in model.wv.key_to_index]
        if len(line_vectors) > 1:
            traces.append(go.Scatter3d(
                x=[v[0] for v in line_vectors],
                y=[v[1] for v in line_vectors],
                z=[v[2] for v in line_vectors],
                mode='lines',
                line=dict(color=hex_colors[i], width=2),
                name=f"Sentence {i+1}",
                showlegend=True
            ))
    return traces

def run(sentences):
    st.subheader("🧭 3D Vector Space View")

    init_session_state(sentences)
    all_options = [f"Sentence {i+1}: {s}" for i, s in enumerate(sentences)]

    with st.expander("🎯 Select Sentences to Show Connection Lines", expanded=True):
        selected_labels = st.multiselect(
            "Choose sentences to visualize:",
            options=all_options,
            default=st.session_state["sentence_picker"],
            key=None
        )
        multiselect_changed("sentence_picker", selected_labels)
        st.markdown(f"✅ Currently selected: **{len(st.session_state['sentence_picker'])} sentence(s)**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Run Visualization", key="run_viz_button_3d"):
                st.session_state["selected_indices_3d"] = [
                    i for i, label in enumerate(all_options) if label in st.session_state["sentence_picker"]
                ]
                st.session_state["trigger_plot_3d"] = True
        with col2:
            if st.button("🔁 Reset Selection", key="reset_viz_button_3d"):
                st.session_state["selected_indices_3d"] = [0, 1]
                st.session_state["sentence_picker"] = [f"Sentence {i+1}: {s}" for i, s in enumerate(sentences[:2])]
                st.session_state["trigger_plot_3d"] = False

    if not st.session_state.get("trigger_plot_3d", False):
        st.warning("⚠️ You've modified your selection. Please click 'Run Visualization' again.")
        return

    with st.spinner("🔄 Rendering 3D Word Embedding Plot..."):
        tokenized_sentences = [simple_preprocess(s) for s in sentences]
        model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
        word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])

        if word_vectors.shape[0] < 3 or word_vectors.shape[1] < 3:
            st.error("❌ Not enough data to perform PCA.")
            return

        reduced_vectors = PCA(n_components=3).fit_transform(word_vectors)
        cmap = plt.get_cmap('tab20', len(tokenized_sentences))
        hex_colors = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255)) for r, g, b, a in [cmap(i) for i in range(len(tokenized_sentences))]]

        word_colors = []
        for word in model.wv.index_to_key:
            for i, sentence in enumerate(tokenized_sentences):
                if word in sentence:
                    word_colors.append(hex_colors[i])
                    break

        fig = go.Figure()
        fig.add_trace(_draw_scatter(reduced_vectors, model, word_colors))
        fig.add_traces(_draw_lines(reduced_vectors, model, tokenized_sentences, hex_colors))

        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title="3D Word Embedding Visualization",
            width=1000, height=900,
            legend=dict(
                title="Selected Sentences",
                orientation="v",    # 垂直排
                x=1.05,             # Legend稍微往右移
                y=1,
                bgcolor="rgba(255,255,255,0.7)",  # 半透明白色背景
                bordercolor="Black",
                borderwidth=1
            )
        )


        st.plotly_chart(fig, use_container_width=True, key=f"view3d_plot_{int(time.time()*1000)}")

    with st.expander("📄 Show Input Sentences", expanded=False):
        for i, sentence in enumerate(sentences, 1):
            st.markdown(f"**Sentence {i}:** {sentence}")
