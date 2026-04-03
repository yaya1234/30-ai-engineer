from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 阿里云配置
API_KEY = ""  # 替换为你的真实 API Key

# 加载文档
loader = TextLoader("data/titanic.txt", encoding="utf-8")
documents = loader.load()

# 分割文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

# 嵌入模型（阿里 text-embedding-v1）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=API_KEY
)

# 向量数据库
vectordb = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
vectordb.persist()

# 检索器
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# 大模型（通义千问）
llm = Tongyi(model="qwen-turbo", dashscope_api_key=API_KEY, temperature=0)

# 提示词模板
template = """你是一个基于文档的问答助手。请根据以下上下文回答用户的问题。
如果上下文没有提供相关信息，就说“根据已有文档，无法回答该问题”。

上下文：
{context}

问题：{question}

答案："""
prompt = ChatPromptTemplate.from_template(template)

# 构建 RAG 链
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 交互式问答
while True:
    query = input("请输入问题（输入 q 退出）: ")
    if query.lower() == 'q':
        break
    result = rag_chain.invoke(query)
    print(f"答案: {result}")