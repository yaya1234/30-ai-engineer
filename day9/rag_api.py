from flask import Flask, request, jsonify
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------- 全局初始化 ----------
# 阿里云 API Key（建议从环境变量读取）
API_KEY = ""  # 替换为你的真实 key

# 加载文档（与之前相同，但只需执行一次）
loader = TextLoader("data/titanic.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=API_KEY
)

vectordb = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

llm = Tongyi(model="qwen-turbo", dashscope_api_key=API_KEY, temperature=0)

template = """你是一个基于文档的问答助手。请根据以下上下文回答用户的问题。
如果上下文没有提供相关信息，就说“根据已有文档，无法回答该问题”。

上下文：
{context}

问题：{question}

答案："""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ---------- Flask 应用 ----------
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Missing "question" field'}), 400
    question = data['question']
    try:
        answer = rag_chain.invoke(question)
        return jsonify({'question': question, 'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  # 使用 5001 端口避免与模型 API 冲突