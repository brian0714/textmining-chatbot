from pdf_context import *
from qa_utils import *
import re

def generate_response(prompt):
    pdf_context = get_pdf_context()
    original_prompt = prompt
    prompt = prompt.strip().lower()

    if prompt not in ["show content", "clustering analysis", "esg analysis"] and "show pdf page" not in prompt:
        return (
            "📝 It looks like your prompt might not match the expected operations.\n\n"
            "💡 Try entering prompts like:\n"
            "- Show content\n"
            "- Show pdf page <num>\n"
            "- Clustering analysis\n"
            "- ESG analysis\n\n"
            "📄 Also, make sure you've uploaded a PDF file first!"
        )

    if not pdf_context:
        return f"Please upload a PDF file to get context."
    elif prompt == "show content":
        # {pdf_context[:1000]}
        return f"""
        🤖 Here's what I found from the uploaded PDF:\n
        {pdf_context}
        ----------------------------------\n
        """
    elif "show pdf page" in prompt:
        match = re.search(r"show pdf page (\d+)", prompt)
        if match:
            page_number = int(match.group(1))
            return get_pdf_context(page=page_number)
        else:
            return "⚠️ Please specify the page number, e.g., `Show PDF page 2`."
    elif prompt == "clustering analysis":

        # "colab code"

        return f"📊 Working on clustering analysis..."
    elif prompt == "esg analysis":

        # "colab code"

        return f"🌱 Working on ESG analysis..."

    # 加一個 fallback return，防止漏掉時回傳 None
    return f"⚠️ Unexpected issue of prompt - ```{original_prompt}```. Please try again."
