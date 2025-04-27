from pdf_context import *
from qa_utils.Word2vec import view_2d, view_3d, skipgram, cbow, negative_sampling
import re

def generate_response(prompt):
    pdf_context = get_pdf_context()
    original_prompt = prompt
    prompt = prompt.strip().lower()

    # 可執行 Word2Vec 子模組對應表
    vector_semantics_tasks = {
        "view2d": (view_2d.run, "🧭 2D Word Embedding Visualization is ready to run. Please provide your input sentences in the UI."),
        # "view3d": (view_3d.run, "📡 3D Word Embedding Visualization is ready to run."),
        # "cbow": (cbow.run, "📘 CBOW model is ready to run."),
        # "skipgram": (skipgram.run, "⚙️ Skip-gram model is ready to run."),
        # "negative sampling": (negative_sampling.run, "🔍 Negative Sampling is ready to run.")
    }

    prompt_lists = [
        "show content",
        "clustering analysis",
        "esg analysis",
        "vector semantics - word2vec",
        "view2d",
        "view3d",
        "cbow",
        "skipgram",
        "negative sampling",
    ]

    if prompt not in prompt_lists and "show pdf page" not in prompt:
        return (
            "📝 It looks like your prompt might not match the expected operations.\n\n"
            "💡 Try entering prompts like:\n"
            "- `Show content`\n"
            "- `Show pdf page <num>`\n"
            "- `Vector Semantics - Word2vec`\n"
            "- `Clustering analysis`\n"
            "- `ESG analysis`\n\n"
            "📄 Also, make sure you've uploaded a PDF file first!"
        )

    if (prompt == "show content" and not pdf_context) or \
       ("show pdf page" in prompt and not pdf_context):
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

    elif prompt == "vector semantics - word2vec":
        return (
            "📊 You're now in the **Vector Semantics - Word2Vec** module!\n\n"
            "You can enter one of the following prompts to run specific visualizations:\n"
            "- `view2d` → 2D Word Embedding Visualization\n"
            "- `view3d` → 3D Word Embedding Visualization\n"
            "- `cbow` → CBOW model explanation or demo\n"
            "- `skipgram` → Skip-gram model explanation or demo\n"
            "- `negative sampling` → Negative Sampling demo\n\n"
            "💡 For example, type `view2d` to run the 2D vector space visualization."
        )

    elif prompt in vector_semantics_tasks:
        st.session_state["pending_vector_task"], message = vector_semantics_tasks[prompt]
        return message

    # Not yet..
    elif prompt == "clustering analysis":

        # "colab code"

        return f"📊 Working on clustering analysis..."
    elif prompt == "esg analysis":

        # "colab code"

        return f"🌱 Working on ESG analysis..."

    # 加一個 fallback return，防止漏掉時回傳 None
    return f"⚠️ Unexpected issue of prompt - ```{original_prompt}```. Please try again."

