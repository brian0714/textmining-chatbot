import streamlit as st
import re
import nltk
import time

# --- 保證 nltk 必要資源 ---
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# --- 停用詞載入 ---
def load_stopwords(filepath='lib/chinese_stopwords.txt'):
    try:
        with open(filepath, encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
    except:
        stopwords = set()
    return list(stopwords)

CHINESE_STOPWORDS = load_stopwords()
ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))

# --- 基礎工具 ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def detect_pdf_language(doc, max_pages=10):
    """簡單偵測 PDF 語言：抽樣前幾頁字數"""
    if not doc:
        return "unknown"

    sample_text = ""
    for page_number, page in enumerate(doc):
        if page_number >= max_pages:
            break
        try:
            sample_text += page.get_text()
        except:
            continue

    chinese_chars = sum(1 for c in sample_text if '\u4e00' <= c <= '\u9fff')
    english_chars = sum(1 for c in sample_text if c.isascii() and c.isalpha())

    if chinese_chars > english_chars:
        return "chinese"
    elif english_chars > chinese_chars:
        return "english"
    else:
        return "unknown"

# --- 中文處理：延遲載入 CKIP ---
def preprocess_text_chinese(text):
    """繁中或簡中處理流程 (清洗 -> 斷詞 -> 去停用詞)，並且延遲載入 CKIP"""
    # ❗延遲載入，只在第一次用到中文時載入模型
    if "ckip_ws_driver" not in st.session_state:
        with st.spinner("🔄 Loading CKIP models (first time using Chinese text)..."):
            from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker

            st.session_state.ckip_ws_driver = CkipWordSegmenter(model="bert-base")
            st.session_state.ckip_pos_driver = CkipPosTagger()
            st.session_state.ckip_ner_driver = CkipNerChunker()
            st.success("✅ CKIP models loaded successfully!")

    ws_driver = st.session_state.ckip_ws_driver

    start_time = time.time()

    # 1. 清理
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\u4e00-\u9fffA-Za-z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 2. CKIP斷詞
    ws = ws_driver([text])[0]
    ws = [re.sub(r'\s+', '', w) for w in ws if w.strip()]

    # 3. 停用詞過濾
    pdf_words = ["None", None, "n", "Col", "Table"]
    self_defined_stopwords = [
        "中", "年", "完成", "共好", "月", "董事", "董事會", "集團", "公司", "目標", "委員會", "兩", "高",
        "主題", "機制", "持續", "提", "提名", "發展", "職場", "參與", "經濟", "核心", "中央", "社會",
        "管理", "相關", "確保", "台灣", "海納", "次", "員工", "全球", "評估", "稽核", "年度", "幸福",
        "共贏", "包容", "單位", "至少", "客戶"
    ]
    all_stopwords = set(CHINESE_STOPWORDS + pdf_words + self_defined_stopwords)
    for w in ws:
        for word in pdf_words:
            if word == None:
                continue
            elif word.lower() in w.lower():
                all_stopwords.add(w)
    ws_filtered = [w for w in ws if w not in all_stopwords]

    elapsed_time = time.time() - start_time
    print(f"Preprocess Chinese text completed in {elapsed_time:.2f} seconds.")
    return ws_filtered

# --- 核心功能 ---
def extract_text_by_page(doc, max_pages=40, skip_pages=[]):
    formatted_full_text = []
    total_items = len(doc)
    total_pages = min(total_items, max_pages)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for page_number, page in enumerate(doc):
        if page_number >= max_pages:
            break
        if int(page_number) + 1 in skip_pages:
            message = f"⏭️ Skip page {page_number + 1}"
            print(message)
            status_text.info(message)
            continue

        try:
            this_text = clean_text(page.get_text())

            tables = page.find_tables()
            for table in tables:
                df = table.to_pandas()
                this_text += "\nTable:\n" + df.to_string() + "\n"

            print(f"Text length in page {page_number+1}: {len(this_text)}")

            formatted_full_text.append({
                "page": page_number + 1,
                "content": this_text
            })

            progress = (page_number + 1) / total_pages
            message = f"Progress: {round(progress*100)}% | Processing {page_number+1}/{total_pages} pages..."
            print(message)
            progress_bar.progress(progress)
            status_text.info(message)

        except Exception as e:
            error_msg = f"(extract_text_by_page) Error processing page {page_number+1}: {e}"
            print(error_msg)
            st.error(error_msg)

    print("Processing complete!")
    progress_bar.progress(1.0)
    status_text.success("✅ PDF processing complete!")

    language = detect_pdf_language(doc)
    st.session_state["pdf_language"] = language
    st.info(f"🌏 Detected PDF language: **{language.upper()}**")

    return formatted_full_text

def get_pdf_context(page="all") -> str:
    if "pdf_text" not in st.session_state:
        return ""

    if page != "all":
        for p in st.session_state["pdf_text"]:
            if p["page"] == page:
                return f"[Page {p['page']}]: {p['content']}"
        return f"Page {page} not found in the PDF."

    return "\n\n".join([f"[Page {p['page']}]: {p['content']}" for p in st.session_state["pdf_text"]])

def preprocess_pdf_sentences(raw_text, tokenize=True):
    if not raw_text or not isinstance(raw_text, str):
        return []

    language = st.session_state.get("pdf_language", "auto")

    results = []
    page_paragraphs = raw_text.split("\n\n")

    for paragraph in page_paragraphs:
        cleaned = re.sub(r"\[Page\s*\d+\]:\s*", "", paragraph).strip()
        if not cleaned:
            continue

        if language == "chinese":
            tokens = preprocess_text_chinese(cleaned)
            if tokens:
                results.append(" ".join(tokens))
        else:
            if tokenize:
                split_sentences = nltk.sent_tokenize(cleaned)
                results.extend([s for s in split_sentences if s.strip()])
            else:
                results.append(cleaned)

    return results
