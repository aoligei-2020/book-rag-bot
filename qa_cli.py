from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Tuple
import json

SYSTEM_PROMPT = """你是“只依据证据作答”的图书问答助手。
规则：
1) 仅使用提供的片段作答；不要使用常识或外部知识。
2) 重要事实（人物名、地名、时间等）必须在片段中出现过，且给出证据编号。
3) 若片段无法支持答案，务必输出“不确定/未在书中找到”。
4) 能够时请摘录原文短句（带引号）并附【编号】。
输出JSON，形如：
{{"answer": "...", "citations": [1,2]}}
"""

QA_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "问题：{question}\n\n片段：\n{context}\n\n请按JSON输出。")
])

def format_docs(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        txt = d.page_content.replace("\n", " ")
        if len(txt) > 700:
            txt = txt[:700] + "…"
        meta = d.metadata or {}
        src = meta.get("source", "")
        lines.append(f"{txt}  (source: {src})")
    return "\n".join(lines)

def retrieve(vs: Chroma, question: str, k: int = 8) -> List[Document]:
    retriever = vs.as_retriever(search_kwargs={"k": k})
    # 新接口：用 invoke() 代替 get_relevant_documents()
    return retriever.invoke(question)

def build_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def robust_answer(llm, question: str, docs: List[Document]) -> Tuple[str, List[int], str]:
    ctx = format_docs(docs) if docs else "(无片段)"
    msg = QA_TEMPLATE.format_messages(question=question, context=ctx)
    raw = llm.invoke(msg).content

    ans, cites = "不确定/未在书中找到", []
    try:
        data = json.loads(raw)
        ans = (data.get("answer") or "").strip()
        cites = data.get("citations") or []
    except Exception:
        pass

    if not cites:
        ans = "不确定/未在书中找到"

    debug = raw
    return ans, cites, debug

# ... 省略上文 import 与前置代码 ...

def sanitize_citations(cites_raw, num_docs: int):
    """把模型返回的 citations 清洗成有效编号列表（1..num_docs）"""
    clean = []
    if not isinstance(cites_raw, list):
        return clean
    for x in cites_raw:
        # 兼容 "【3】"、"3."、"3 "、"①3" 等奇怪格式：提取数字
        s = str(x)
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            continue
        i = int(digits)
        if 1 <= i <= num_docs:
            clean.append(i)
    # 去重并保持顺序
    seen = set()
    uniq = []
    for i in clean:
        if i not in seen:
            seen.add(i)
            uniq.append(i)
    return uniq

def print_evidence(citations, docs):
    if not citations:
        print("证据: （未命中足够证据）")
        return
    marks = "、".join([f"" for i in citations])
    print(f"证据: {marks}")
    # 同时把每条证据的原文片段打印一小段，便于核对
    for i in citations:
        d = docs[i-1]
        snippet = d.page_content.replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:180] + "…"
        src = (d.metadata or {}).get("source", "")
        print(f"  {snippet} (source: {src})")

def robust_answer(llm, question: str, docs: List[Document]):
    ctx = format_docs(docs) if docs else "(无片段)"
    msg = QA_TEMPLATE.format_messages(question=question, context=ctx)
    raw = llm.invoke(msg).content

    ans = "不确定/未在书中找到"
    cites = []
    try:
        data = json.loads(raw)
        ans = (data.get("answer") or "").strip()
        cites = sanitize_citations(data.get("citations"), len(docs))
    except Exception:
        pass

    if not cites:
        ans = "不确定/未在书中找到"
    return ans, cites, raw

if __name__ == "__main__":
    vs = Chroma(
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory="storage"
    )
    llm = build_llm()

    print("书籍问答已就绪。输入你的问题，或输入 /exit 退出。")
    while True:
        q = input("Q> ").strip()
        if not q or q.lower() in {"/exit", "exit", "quit"}:
            break

        docs = retrieve(vs, q, k=8)
        answer, citations, debug = robust_answer(llm, q, docs)

        print("\nA>", answer)
        print_evidence(citations, docs)

        # 调试：遇到奇怪输出时，先打开下面两行看看返回了什么
        # print("\n[检索片段]\n", format_docs(docs))
        # print("\n[模型原始输出]\n", debug)
        print()
