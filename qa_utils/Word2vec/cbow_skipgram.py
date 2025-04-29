import streamlit as st
import nltk
import pandas as pd
import plotly.express as px
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.decomposition import PCA
from pdf_context import preprocess_pdf_sentences
from ui_utils import display_pretty_table

# --- 保證 nltk 資料有載好 ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown', quiet=True)

def init_session_state():
    defaults = {
        "vector_size": 100,
        "window_size": 5,
        "min_count": 1,
        "workers": 4,
        "sg": 1,
        "user_sentences": "",
        "query_word": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

def clean_and_tokenize(sentences):
    """Remove stopwords and tokenize each sentence."""
    tokenized = []
    for sentence in sentences:
        if sentence.strip():
            cleaned = remove_stopwords(sentence)
            tokens = simple_preprocess(cleaned)
            if tokens:
                tokenized.append(tokens)
    return tokenized

def train_word2vec(tokenized_sentences):
    if not tokenized_sentences:
        st.error("❌ No tokenized sentences to train on. Please input some data.")
        st.stop()

    model = Word2Vec(
        vector_size=st.session_state.vector_size,
        window=st.session_state.window_size,
        min_count=st.session_state.min_count,
        workers=st.session_state.workers,
        sg=st.session_state.sg
    )
    model.build_vocab(tokenized_sentences)
    model.train(tokenized_sentences, total_examples=len(tokenized_sentences), epochs=model.epochs)
    return model

def plot_embeddings(model, query_words):
    if not query_words:
        st.warning("Please input at least one word to visualize.")
        return

    vectors = []
    labels = []
    for word in query_words:
        if word in model.wv:
            vectors.append(model.wv[word])
            labels.append(word)

    if len(vectors) < 2:
        st.warning("Not enough words found in vocabulary for plotting.")
        return

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    df = pd.DataFrame({
        'X': reduced[:, 0],
        'Y': reduced[:, 1],
        'Word': labels
    })

    fig = px.scatter(df, x='X', y='Y', text='Word', title="Word Embeddings Visualization")
    st.plotly_chart(fig, use_container_width=True)

def run(sentences=None, source="manual"):
    st.markdown("---")
    st.subheader("🧠 CBOW and Skip-Gram Word2Vec Trainer")

    init_session_state()

    with st.expander("⚙️ Model Settings", expanded=True):
        st.session_state.vector_size = st.slider("Vector Size", 50, 300, 100, step=10)
        st.session_state.window_size = st.slider("Window Size", 2, 10, 5)
        st.session_state.min_count = st.slider("Min Word Count", 1, 5, 1)
        st.session_state.workers = st.slider("Number of Workers", 1, 8, 4)
        model_type = st.radio("Training Algorithm", ["Skip-Gram", "CBOW"], index=0)
        st.session_state.sg = 1 if model_type == "Skip-Gram" else 0

    st.markdown("---")

    # --- processed_sentences 準備 ---
    if source == "pdf":
        if sentences:
            processed_sentences = preprocess_pdf_sentences(raw_text=sentences, tokenize=True)
        else:
            st.error("❌ No PDF sentences loaded.")
            st.stop()
    elif source == "manual":
        if sentences:
            processed_sentences = sentences
        else:
            processed_sentences = [" ".join(sent) for sent in nltk.corpus.brown.sents()]
    else:
        st.error(f"❌ Unknown source '{source}'.")
        st.stop()

    # --- 預先顯示目前的資料概況 ---
    tokenized_preview = clean_and_tokenize(processed_sentences)
    total_sentences = len(tokenized_preview)
    total_tokens = sum(len(tokens) for tokens in tokenized_preview)
    st.info(f"ℹ️ Prepared {total_sentences} tokenized sentences, total {total_tokens} tokens.")

    if st.button("🚀 Train Word2Vec Model"):
        tokenized_sentences = clean_and_tokenize(processed_sentences)

        if not tokenized_sentences:
            st.error("❌ No valid tokenized sentences found after preprocessing.")
            st.stop()

        model = train_word2vec(tokenized_sentences)

        # --- 判斷來源顯示不同 success 訊息 ---
        if source == "pdf":
            st.success("✅ Model trained successfully from **PDF content**!")
        elif sentences:
            st.success("✅ Model trained successfully from **manual input**!")
        else:
            st.success("✅ Model trained successfully from **default Brown corpus**!")

        st.markdown("---")

        with st.expander("🔍 Query Word", expanded=True):
            st.session_state.query_word = st.selectbox(
                "Choose a word to find similar words:",
                options=model.wv.index_to_key if model.wv.index_to_key else ["No words available"]
            )

        if st.session_state.query_word in model.wv:
            st.markdown(f"### 🔥 Similar Words to `{st.session_state.query_word}`:")
            similar_words = model.wv.most_similar(st.session_state.query_word, topn=5)
            df = pd.DataFrame(similar_words, columns=["Word", "Similarity"]).reset_index(drop=True)
            display_pretty_table(df)
        else:
            st.warning(f"Word '{st.session_state.query_word}' not found in vocabulary.")

        st.markdown("---")

        if st.button("🔎 Show Embedding Visualization"):
            sample_words = [st.session_state.query_word] + [word for word, _ in model.wv.most_similar(st.session_state.query_word, topn=5)]
            plot_embeddings(model, sample_words)
