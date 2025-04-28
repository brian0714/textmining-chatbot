import streamlit as st
import re
import nltk

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def extract_text_by_page(doc, max_pages=40, skip_pages=[]):
    formatted_full_text = []
    total_items = len(doc)
    total_pages = min(len(doc), max_pages)

    for page_number, page in enumerate(doc):
        if page_number >= max_pages:
            break
        if int(page_number) + 1 in skip_pages:
            print(f"Skip page {page_number+1}")
            continue

        try:
            this_text = clean_text(page.get_text())

            # Extract tables
            tables = page.find_tables()
            for table in tables:
                df = table.to_pandas()
                this_text += "\nTable:\n" + df.to_string() + "\n"
            print(f"Text length in page {page_number+1}: {len(this_text)}")

            formatted_full_text.append({
                "page": page_number + 1,
                "content": this_text
            })

            # Update progress
            progress = (page_number + 1) / total_pages
            print(f"Progress: {round(progress*100)}%")
            print(f"Processing {page_number + 1}/{total_pages} pages with document (max_pages:{max_pages})...")

        except Exception as e:
            print(f"(extract_text_by_page) Error processing page {page}: {e}")

    print("Processing complete!")

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

def preprocess_pdf_sentences(raw_text):
    """
    Preprocesses PDF text that contains page markers like [Page 1]:.

    Args:
        raw_text (str): Raw extracted text from PDF, formatted like "[Page 1]: content\n\n[Page 2]: content".

    Returns:
        List[List[str]]: A list of tokenized sentences (each sentence is a list of words).
    """
    if not raw_text or not isinstance(raw_text, str):
        return []

    tokenized_sentences = []

    # Split by two newlines (page break)
    page_paragraphs = raw_text.split("\n\n")

    for paragraph in page_paragraphs:
        # Remove the [Page x]: pattern at the beginning
        cleaned = re.sub(r"\[Page\s*\d+\]:\s*", "", paragraph).strip()
        if cleaned:
            sentences = nltk.sent_tokenize(cleaned)
            for sent in sentences:
                words = nltk.word_tokenize(sent)
                if words:
                    tokenized_sentences.append(words)

    return tokenized_sentences
