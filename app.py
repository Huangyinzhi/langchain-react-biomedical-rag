from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from agent.react_agent import ReactAgent

st.set_page_config(page_title="慢病随访智能助手", page_icon="🩺")
st.title("🩺 慢病随访智能助手")
st.caption("基于 LangGraph ReAct‑Agent + RAG 心血管慢病评估系统")
st.divider()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

prompt = st.chat_input()
if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})
    response_messages: list[str] = []
    with st.spinner("AI正在思考，检索慢病知识库并评估数据…"):
        res_stream = st.session_state["agent"].execute_stream(prompt)

        def stream_generator(generator, cache_list):
            full_text = ""
            for chunk in generator:
                full_text += chunk
                yield chunk
            cache_list.append(full_text)

        st.chat_message("assistant").write_stream(stream_generator(res_stream, response_messages))
        st.session_state["messages"].append({"role": "assistant", "content": "".join(response_messages)})
        st.rerun()
