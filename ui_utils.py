import fitz  # PyMuPDF
import streamlit as st
from pdf_context import *

# pdf upload section
def pdf_upload_section():
    with st.expander("📄 Upload a PDF file", expanded=True):
        uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], label_visibility="collapsed")

        # 若已解析 pdf 就不要重複執行
        if uploaded_file and "pdf_text" not in st.session_state:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            extracted = extract_text_by_page(doc, max_pages=len(doc))
            st.session_state["pdf_text"] = extracted
            st.success("✅ PDF uploaded and parsed successfully!")

        # Clear button
        if "pdf_text" in st.session_state:
            if st.button("🗑️ Clear PDF"):
                del st.session_state["pdf_text"]
                st.rerun()

# alert section
def show_dismissible_alert(key: str, text: str, alert_type="warning"):
    colors = {
        "warning": {"bg": "#FFF3CD", "border": "#FFA502"},
        "info": {"bg": "#D1ECF1", "border": "#0C5460"},
        "success": {"bg": "#D4EDDA", "border": "#28A745"},
        "danger": {"bg": "#F8D7DA", "border": "#DC3545"},
    }

    style = colors.get(alert_type, colors["warning"])

    if f"hide_{key}" not in st.session_state:
        st.session_state[f"hide_{key}"] = False

    if not st.session_state[f"hide_{key}"]:

        # 將 ❌ 放在 alert 裡面
        close = st.button("❌", key=f"close_{key}", help="Close this alert")

        if close:
            st.session_state[f"hide_{key}"] = True
            return  # 提前結束，不顯示 alert

        # 顯示 alert 本體
        st.markdown(
            f"""
            <div style="padding:10px 10px 10px 10px;background-color:{style['bg']};
                        border-left:6px solid {style['border']};
                        margin-bottom:10px;position:relative;border-radius:5px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>{text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
