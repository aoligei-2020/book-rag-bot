from pathlib import Path
import sys
from typing import Iterable, List

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import tiktoken

# ===== 可调整参数 =====
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_TOKENS_LIMIT = 200_000        # 每批次总 token 上限（小于 300k 的接口硬上限，留余量）
EMBED_MODEL = "text-embedding-3-small"
PERSIST_DIR = "storage"
# =====================

def load_docs(path: str) -> List[Document]:
    p = Path(path)
    if p.suffix.lower() in [".txt", ".md"]:
        return TextLoader(str(p), encoding="utf-8").load()
    elif p.suffix.lower() == ".pdf":
        return PyPDFLoader(str(p)).load()
    else:
        raise ValueError("只支持 .txt/.md/.pdf")

def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    return splitter.split_documents(docs)

def batch_by_token(docs: List[Document],
                   enc,
                   limit: int = BATCH_TOKENS_LIMIT) -> Iterable[List[Document]]:
    batch, tok_sum = [], 0
    for d in docs:
        # 估算单条的 token 数（注意：embedding 模型用 cl100k_base 编码器通常是合理的近似）
        t = len(enc.encode(d.page_content))
        # 保险：如果单条本身过大（不太可能，因为我们切过块），再粗暴二次截断
        if t > limit:
            # 进一步硬切，以免任何一条超限
            text = d.page_content
            # 粗略等比例截断到 ~limit 的 90%
            cut = enc.decode(enc.encode(text)[: int(limit*0.9) ])
            d = Document(page_content=cut, metadata=d.metadata)
            t = len(enc.encode(d.page_content))

        if batch and tok_sum + t > limit:
            yield batch
            batch, tok_sum = [d], t
        else:
            batch.append(d); tok_sum += t

    if batch:
        yield batch

def main():
    if len(sys.argv) < 2:
        print("用法: python ingest.py data/book.txt")
        sys.exit(1)

    path = sys.argv[1]
    print(f"[1/4] 读取文档：{path}")
    raw_docs = load_docs(path)

    print(f"[2/4] 切分文档（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")
    chunks = split_docs(raw_docs)
    print(f"    -> 共 {len(chunks)} 个切片")

    print(f"[3/4] 初始化向量库与嵌入模型（{EMBED_MODEL}）")
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vs = Chroma(embedding_function=embeddings, persist_directory=PERSIST_DIR)

    print("[4/4] 分批写入（按总 token 限制）")
    enc = tiktoken.get_encoding("cl100k_base")

    total, batch_id = 0, 0
    for batch in batch_by_token(chunks, enc, BATCH_TOKENS_LIMIT):
        batch_id += 1
        try:
            vs.add_documents(batch)
            vs.persist()  # 每批持久化，进度可保留
            total += len(batch)
            print(f"  - 批次 {batch_id}: 写入 {len(batch)} 条（累计 {total}）")
        except Exception as e:
            print(f"  ! 批次 {batch_id} 失败：{e}")
            print("    尝试减小 BATCH_TOKENS_LIMIT 或 CHUNK_SIZE 后重试")
            raise

    print(f"完成：共写入 {total} 个切片 → {PERSIST_DIR}/")

if __name__ == "__main__":
    main()
