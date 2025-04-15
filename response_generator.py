import fitz  # PyMuPDF
import pymupdf
import streamlit as st
import re

def load_pdf(pdf_path):
    """Loads a PDF file using PyMuPDF.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A PyMuPDF document object if successful, None otherwise.
    """
    try:
        doc = pymupdf.open(pdf_path, filetype="pdf")
        return doc
    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_path}'")
        return None
    except Exception as e:
        print(f"An error occurred while loading the PDF: {e}")
        return None

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
            print(f"Progress: {round(progress, 2)}")
            print(f"Processing {page_number + 1}/{total_pages} pages with document (max_pages:{max_pages})...")

        except Exception as e:
            print(f"(extract_text_by_page) Error processing page {page}: {e}")

    print("Processing complete!")

    return formatted_full_text

def pdf_upload_section():
    uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
    if uploaded_file:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        # if not st.session_state["pdf_text"]:
        extracted = extract_text_by_page(doc)
        st.session_state["pdf_text"] = extracted
        st.success("✅ PDF uploaded and parsed successfully!")
        if st.button("🗑️ Clear PDF"):
            del st.session_state["pdf_text"]
            st.experimental_rerun()

def get_pdf_context():
    if "pdf_text" in st.session_state:
        return "\n\n".join([f"[Page {p['page']}]: {p['content'][:300]}..." for p in st.session_state["pdf_text"]])
    return ""

def generate_response(prompt):
    pdf_context = get_pdf_context()

    if prompt not in ["Show content", "Clustering analysis", "ESG analysis"]:
        return (
            "📝 It looks like your prompt might not match the expected operations.\n\n"
            "💡 Try entering prompts like:\n"
            "- Show content\n"
            "- Clustering analysis\n"
            "- ESG analysis\n\n"
            "📄 Also, make sure you've uploaded a PDF file first!"
        )

    if pdf_context and prompt == "Show content":
        # {pdf_context[:1000]}
        return f"""
        🤖 Here's what I found from the uploaded PDF:\n
        {pdf_context}
        ----------------
        (Enter a more specific instructing prompt for better results!)
        """
    elif not pdf_context and prompt == "Show content":
        return f"Please upload a PDF file to get context."
    elif prompt == "Clustering analysis":
        return f"Working on clusetering analysis..."
    elif prompt == "ESG analysis":

        # "colab code"

        return f"Working on ESG analysis..."




