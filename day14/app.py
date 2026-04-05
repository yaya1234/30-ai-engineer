import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.chains import RetrievalQA

st.set_page_config(page_title="学霸助手 - 考证资料库", layout="wide")
st.title("📚 学霸助手 · 专属资料库（RAG）")

# 初始化 session 状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 侧边栏：上传文档
with st.sidebar:
    st.header("📂 上传学习资料")
    uploaded_file = st.file_uploader("支持 TXT、PDF、Word", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        # 加载并处理文档
        loader = TextLoader(tmp_path, encoding="utf-8")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        # 创建向量库（使用阿里云通义嵌入）
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"]
        )
        st.session_state.vectorstore = Chroma.from_documents(texts, embeddings)
        os.unlink(tmp_path)
        st.success("资料已收录！")

# 聊天界面
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("输入你的问题，例如：行政强制措施有哪些种类？"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore is None:
        response = "请先在上传学习资料。"
    else:
        # 构建 RAG 链
        llm = Tongyi(model="qwen-turbo", dashscope_api_key=st.secrets["DASHSCOPE_API_KEY"])
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = qa_chain.run(prompt)
        # 添加引用（简单拼接）
        response += "\n\n> 来源：你上传的资料"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})