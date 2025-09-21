from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
import sys

def load_docs(path: str):
    p = Path(path)
    if p.suffix.lower() in [".txt", ".md"]:
        return TextLoader(str(p), encoding="utf-8").load()
    elif p.suffix.lower() == ".pdf":
        return PyPDFLoader(str(p)).load()
    else:
        raise ValueError("只支持 .txt/.md/.pdf")

def main():
    if len(sys.argv) < 2:
        print("用法: python ingest.py data/book.txt")
        sys.exit(1)

    docs = load_docs(sys.argv[1])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="storage"
    )
    vs.persist()
    print(f"完成：{len(chunks)} 个切片已写入 storage/")

if __name__ == "__main__":
    main()
