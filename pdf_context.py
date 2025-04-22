import streamlit as st
import re

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
