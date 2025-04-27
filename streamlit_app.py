import json
import streamlit as st
from openai import OpenAI
import time
import requests
from db_utils import init_db, get_user_profile, save_user_profile
from qa_utils.Word2vec import view_2d, view_3d, skipgram, cbow, negative_sampling
from ui_utils import *
from pdf_context import *
from response_generator import generate_response

placeholderstr = "Please input your command"
# user_name = "Brian"
# user_image = "https://www.w3schools.com/howto/img_avatar.png"

def stream_data(stream_str):
    for word in stream_str.split(" "):
        yield word + " "
        time.sleep(0.15)

def is_valid_image_url(url):
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200 and 'image' in response.headers["Content-Type"]:
            return True
        else:
            return False
    except:
        return False

# 讀取 JSON 檔案中的範例句子
def load_example_from_json(json_path, key):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key, "")  # 如果找不到key就回傳空字串

def main():
    st.set_page_config(
        page_title='K-Assistant - The Residemy Agent',
        layout='wide',
        initial_sidebar_state='auto',
        menu_items={
            'Get Help': 'https://streamlit.io/',
            'Report a bug': 'https://github.com',
            'About': 'About your application: **Hello world**'
            },
        page_icon="img/favicon.ico"
    )

    # Get User Profile from db
    init_db()
    profile = get_user_profile()

    if "user_name" not in st.session_state:
        st.session_state["user_name"] = profile["user_name"] if profile else "Brian"
    if "user_image" not in st.session_state:
        st.session_state["user_image"] = profile["user_image"] if profile else "https://www.w3schools.com/howto/img_avatar.png"

    # Show title and description.
    user_name = st.session_state["user_name"]
    user_image = st.session_state["user_image"]
    st.title(f"💬 {user_name}'s Chatbot")

    # Left side bar
    with st.sidebar:
        st_c_1 = st.container(border=True)
        with st_c_1:
            if user_image:
                if is_valid_image_url(user_image):
                    st.image(user_image)
                else:
                    # st.warning("⚠️ Invalid avatar URL. Showing default image.")
                    # show_dismissible_alert("⚠️ Invalid avatar URL. Showing default image.<br>Image Ref: https://unsplash.com/", alert_type="warning")
                    show_dismissible_alert(
                        "avatar_warning",
                        "⚠️ Invalid avatar URL.<br>Showing default image.<br>Image Ref: <a href='https://unsplash.com/' target='_blank'>https://unsplash.com/</a>",
                        alert_type="warning"
                    )
                    st.image("https://www.w3schools.com/howto/img_avatar.png")
            else:
                st.image("https://www.w3schools.com/howto/img_avatar.png")

        st.markdown("---")

        # radio expander (radio button 格式)
        # with st.expander("📦 Vector Semantics - Word2vec", expanded=False):
        #     option = st.radio(
        #         "Select a function:",
        #         ["Vector space - 2D View", "Vector space - 3D View", "SKIP-GRAM", "CBOW", "Negative Sampling"],
        #         index=0
        #     )
        #     st.session_state["selected_vector_task"] = option

        with st.expander("📦 Vector Semantics - Word2vec", expanded=False):
            if st.button("🧭 Vector space - 2D View"):
                st.session_state["vector_task"] = view_2d.run

            if st.button("🧭 Vector space - 3D View"):
                st.session_state["vector_task"] = view_3d.run

            # if st.button("⚙️ SKIP-GRAM"):
            #     st.session_state["vector_task"] = 1#skipgram.run

            # if st.button("📘 CBOW"):
            #     st.session_state["vector_task"] = 1#cbow.run

            # if st.button("🔍 Negative Sampling"):
            #     st.session_state["vector_task"] = 1#negative_sampling.run

        st.markdown("---")
        # st.write("🌐 Language")
        selected_lang = st.selectbox("🌐 Language", ["English", "繁體中文"], index=1)
        st.session_state['lang_setting'] = selected_lang

        # st.header("🧑‍💻 Profile Settings")
        with st.expander("🧑‍💻 Profile Settings", expanded=False):
            with st.form(key="profile_form"):
                new_name = st.text_input("User Name", value=st.session_state["user_name"])
                new_image = st.text_input("Avatar Image URL", value=st.session_state["user_image"])
                submitted = st.form_submit_button("💾 Save Profile")

                if submitted:
                    save_user_profile(new_name, new_image)
                    st.session_state["user_name"] = new_name
                    st.session_state["user_image"] = new_image
                    st.success("Profile saved! Please refresh to see changes.")
                    st.rerun()

    st_c_chat = st.container(border=True)
    # pdf upload section
    pdf_upload_section()

    # vector task section
    if "vector_task" in st.session_state and callable(st.session_state["vector_task"]):
        st.markdown("## 🧠 Provide your own sentences for Word2Vec")

        # --- User input block ---
        # 初始化 session_state
        if "user_input_text" not in st.session_state:
            st.session_state["user_input_text"] = ""

        # 按鈕：載入範例
        if st.button("🔖 Load Example Sentences"):
            example_text = load_example_from_json("db/examples.json", "vector semantic example")
            st.session_state["user_input_text"] = example_text

        # 使用者輸入區塊，綁定 session_state
        user_input_text = st.text_area(
            label="Enter sentences (one per line):",
            value=st.session_state["user_input_text"],
            height=300,
            placeholder="Type one sentence per line...\nExample:\nThe food is fresh and safe.\nWe promote energy saving."
        )

        # 每次文字框更新，也同步到 session_state
        st.session_state["user_input_text"] = user_input_text

        # 按鈕：執行向量任務
        if st.button("🚀 Run Vector Task"):
            # 若有input, 分割成 list of sentences
            if user_input_text.strip():
                input_sentences = [line.strip() for line in user_input_text.splitlines() if line.strip()]
                st.session_state["vector_task"](sentences=input_sentences)
            # 無input, 則提示使用者輸入
            else:
                st.warning("⚠️ Please enter some sentences before running the vector task.")

    # chat section
    if "messages" not in st.session_state:
        st.session_state.messages = []
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                if user_image:
                    st_c_chat.chat_message(msg["role"],avatar=user_image).markdown((msg["content"]))
                else:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            elif msg["role"] == "assistant":
                st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))
            else:
                try:
                    image_tmp = msg.get("image")
                    if image_tmp:
                        st_c_chat.chat_message(msg["role"],avatar=image_tmp).markdown((msg["content"]))
                except:
                    st_c_chat.chat_message(msg["role"]).markdown((msg["content"]))

    # Chat function section (timing included inside function)
    def chat(prompt: str):
        if user_image and is_valid_image_url(user_image):
            chat_user_image = user_image
        else:
            chat_user_image = "https://www.w3schools.com/howto/img_avatar.png"
        st_c_chat.chat_message("user", avatar=chat_user_image).write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call generate_response function
        response = generate_response(prompt)
        # response = f"You type: {prompt}"

        st.session_state.messages.append({"role": "assistant", "content": response})
        st_c_chat.chat_message("assistant").write_stream(stream_data(response))


    if prompt := st.chat_input(placeholder=placeholderstr, key="chat_bot"):
        chat(prompt)

    # 將 pending task 指令變成正式指令觸發 rerun
    if "pending_vector_task" in st.session_state:
        st.session_state["vector_task"] = st.session_state["pending_vector_task"]
        del st.session_state["pending_vector_task"]
        st.rerun()  # 🔁 強制 rerun 以觸發 render
    # if "pending_vector_task" in st.session_state:
    #     if "vector_task" not in st.session_state:
    #         st.session_state["vector_task"] = st.session_state["pending_vector_task"]
    #         del st.session_state["pending_vector_task"]
    #         st.query_params(task="set")  # 觸發 UI state 標記，不 rerun
    #     else:
    #         pass  # 已設定，什麼都不做，不要再 rerun

if __name__ == "__main__":
    main()
