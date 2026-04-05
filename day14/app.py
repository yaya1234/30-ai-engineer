import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量（本地开发用）
load_dotenv()

# 文档加载器
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_classic.chains import RetrievalQA

st.set_page_config(page_title="学霸助手 - 考证资料库", layout="wide")
st.title("📚 学霸助手 · 专属资料库（RAG）")

# 初始化 session 状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 获取 API Key：优先从环境变量，其次从 st.secrets（云端）
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["DASHSCOPE_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("未设置 DASHSCOPE_API_KEY，请在环境变量或 Streamlit Secrets 中配置")
    st.stop()

# 侧边栏：上传文档
with st.sidebar:
    st.header("📂 上传学习资料")
    uploaded_file = st.file_uploader("支持 TXT、PDF、Word", type=["txt", "pdf", "docx"])
    if uploaded_file is not None:
        # 保存临时文件
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        # 根据扩展名选择加载器
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

        # 分割文本
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # 创建向量库
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=api_key
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
        response = "请先上传学习资料。"
    else:
        llm = Tongyi(model="qwen-turbo", dashscope_api_key=api_key)
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = qa_chain.run(prompt)
        response += "\n\n> 来源：你上传的资料"

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})