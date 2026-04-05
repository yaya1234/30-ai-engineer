import pandas as pd
import joblib
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent

# ---------- 阿里云配置 ----------
API_KEY = ""  # 替换为你的真实 key
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

llm = ChatOpenAI(
    model="qwen-turbo",
    openai_api_key=API_KEY,
    openai_api_base=BASE_URL,
    temperature=0
)

# ---------- 加载模型预测工具 ----------
model = joblib.load("../day7/titanic_rf_best.pkl")  # 确保路径正确

# 特征顺序（与训练时完全一致）
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']

def predict_survival(pclass: int, sex: str, age: int) -> str:
    """预测乘客生还概率。参数：pclass(1/2/3), sex(male/female), age(数字)"""
    sex_encoded = 1 if sex.lower() == "female" else 0
    # 使用默认值填充其他特征（简化，实际可根据需要调整）
    input_data = {
        'Pclass': pclass,
        'Sex': sex_encoded,
        'Age': age,
        'SibSp': 0,
        'Parch': 0,
        'Fare': 32,
        'Embarked': 0,      # S港口编码为0
        'FamilySize': 1,
        'Title': 0          # Mr编码为0
    }
    X = pd.DataFrame([input_data])[features]
    prob = model.predict_proba(X)[0][1]
    survived = model.predict(X)[0]
    return f"预测结果：{'生还' if survived else '未生还'}，概率 {prob:.2f}"

@tool
def predict_survival_tool(pclass: int, sex: str, age: int) -> str:
    """调用模型预测工具。参数：pclass(1/2/3), sex(male/female), age(数字)"""
    return predict_survival(pclass, sex, age)

# ---------- 加载 RAG 工具 ----------
loader = TextLoader("../day9/data/titanic.txt", encoding="utf-8")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)
embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=API_KEY)
vectordb = Chroma.from_documents(texts, embeddings, persist_directory="../day9/chroma_db")
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

template = """你是一个基于文档的问答助手。请根据以下上下文回答用户的问题。
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

@tool
def ask_document(question: str) -> str:
    """基于文档回答问题。参数：question(用户的问题)"""
    return rag_chain.invoke(question)

tools = [predict_survival_tool, ask_document]

# ---------- 使用预置的 ReAct Agent ----------
agent_executor = create_react_agent(llm, tools)

while True:
    user_input = input("请输入问题（输入 q 退出）: ")
    if user_input.lower() == 'q':
        break
    for chunk in agent_executor.stream(
        {"messages": [("user", user_input)]},
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()