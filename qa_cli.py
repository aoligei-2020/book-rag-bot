from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser

SYSTEM_PROMPT = """你是图书问答助手。
仅基于提供的检索片段回答；如果片段不足，请回答“不确定/未在书中找到”。"""

QA_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "问题：{question}\n\n相关片段：\n{context}\n\n请作答：")
])

def format_docs(docs):
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(f"【{i}】{d.page_content[:500]}")
    return "\n".join(lines)

def build_chain():
    vs = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="storage"
    )
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    chain = (
        {"question": RunnablePassthrough(),
         "context": retriever | format_docs}
        | QA_TEMPLATE
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    chain = build_chain()
    print("书籍问答已就绪。输入你的问题，或输入 /exit 退出。")
    while True:
        q = input("Q> ").strip()
        if not q or q.lower() in {"/exit", "exit", "quit"}:
            break
        ans = chain.invoke(q)
        print("\nA>", ans, "\n")
