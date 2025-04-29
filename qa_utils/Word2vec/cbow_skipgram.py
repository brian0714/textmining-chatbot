import streamlit as st
import nltk
import pandas as pd
import plotly.express as px
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.decomposition import PCA
from pdf_context import preprocess_pdf_sentences

# Ensure necessary NLTK resources are downloaded
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
        if key not in st.session_state:
            st.session_state[key] = value

def get_training_sentences(source="manual"):
    if source == "pdf" and "pdf_text" in st.session_state and st.session_state.pdf_text:
        sentences = preprocess_pdf_sentences(raw_text=st.session_state.pdf_text, tokenize=True)
    else:
        sentences = [" ".join(sent) for sent in nltk.corpus.brown.sents()]

    if st.session_state.user_sentences.strip():
        sentences += st.session_state.user_sentences.strip().split("\n")

    return [simple_preprocess(remove_stopwords(sentence)) for sentence in sentences if sentence.strip()]

def train_word2vec(tokenized_sentences):
    model = Word2Vec(
        tokenized_sentences,
        vector_size=st.session_state.vector_size,
        window=st.session_state.window_size,
        min_count=st.session_state.min_count,
        workers=st.session_state.workers,
        sg=st.session_state.sg
    )
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
    st.subheader("🧠 CBOW and Skip-Gram Word2Vec Trainer")

    init_session_state()

    with st.expander("⚙️ Model Settings", expanded=True):
        st.session_state.vector_size = st.slider("Vector Size", 50, 300, 100, step=10)
        st.session_state.window_size = st.slider("Window Size", 2, 10, 5)
        st.session_state.min_count = st.slider("Min Word Count", 1, 5, 1)
        st.session_state.workers = st.slider("Number of Workers", 1, 8, 4)
        model_type = st.radio("Training Algorithm", ["Skip-Gram", "CBOW"], index=0)
        st.session_state.sg = 1 if model_type == "Skip-Gram" else 0

    with st.expander("📄 Data Source", expanded=True):
        st.session_state.user_sentences = st.text_area("Add more sentences (one per line)", value=st.session_state.user_sentences, height=200)

    st.markdown("---")

    if sentences is None:
        tokenized_sentences = get_training_sentences(source=source)
    else:
        if source == "pdf":
            processed = preprocess_pdf_sentences(raw_text=sentences, tokenize=True)
            tokenized_sentences = [simple_preprocess(remove_stopwords(sentence)) for sentence in processed if sentence.strip()]
        else:
            tokenized_sentences = [simple_preprocess(remove_stopwords(sentence)) for sentence in sentences if sentence.strip()]

    model = train_word2vec(tokenized_sentences)
    st.success("✅ Model trained successfully!")

    st.markdown("---")

    with st.expander("🔍 Query Word", expanded=True):
        st.session_state.query_word = st.selectbox("Choose a word to find similar words:", model.wv.index_to_key)

    if st.session_state.query_word in model.wv:
        st.markdown(f"### 🔥 Similar Words to **{st.session_state.query_word}**:")
        similar_words = model.wv.most_similar(st.session_state.query_word, topn=5)
        st.table(pd.DataFrame(similar_words, columns=["Word", "Similarity"]))
    else:
        st.warning(f"Word '{st.session_state.query_word}' not found in vocabulary.")

    st.markdown("---")

    if st.button("🔎 Show Embedding Visualization"):
        sample_words = [st.session_state.query_word] + [word for word, _ in model.wv.most_similar(st.session_state.query_word, topn=5)]
        plot_embeddings(model, sample_words)
