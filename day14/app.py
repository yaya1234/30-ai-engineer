import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="学霸助手 - 考证资料库", layout="wide")
st.title("📚 学霸助手 · 专属资料库（纯检索版）")
st.markdown("上传资料后，输入问题即可从资料中检索相关片段。无需 API Key，完全免费。")

# 初始化 session
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 加载嵌入模型
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = load_embeddings()

# 侧边栏上传文档
with st.sidebar:
    st.header("📂 上传学习资料")
    uploaded_file = st.file_uploader("支持 TXT、PDF、Word", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            if suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif suffix == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                st.error("不支持的文件格式")
                st.stop()
            documents = loader.load()
        except Exception as e:
            st.error(f"文档加载失败: {str(e)}")
            os.unlink(tmp_path)
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        st.session_state.vectorstore = FAISS.from_documents(texts, embeddings)
        os.unlink(tmp_path)
        st.success(f"已收录 {len(documents)} 页内容，可开始提问。")

# 显示聊天历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 处理用户提问
if prompt := st.chat_input("输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore is None:
        response = "请先上传学习资料。"
    else:
        # 方法1：使用相似性搜索（兼容旧版LangChain）
        docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
        if not docs:
            response = "未找到相关内容。"
        else:
            response = "📖 找到以下相关内容：\n\n"
            for i, doc in enumerate(docs):
                response += f"**片段 {i+1}:**\n{doc.page_content}\n\n"
            response += "> 以上内容摘自你上传的资料。"
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})