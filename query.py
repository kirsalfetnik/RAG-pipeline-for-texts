#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import List
from fpdf import FPDF
from datetime import datetime
import re

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI 
#from langchain_community.chat_models import ChatOpenAI
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

STORE_DIR = Path("vector_store")
DEFAULT_TOP_K = 6

def sanitize(text: str) -> str:
    text = (
        text
        .replace("≥", ">=")
        .replace("≤", "<=")
        .replace("…", "...")
        .replace("–", "-")    
        .replace("—", "--")   
        .replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("↓", "v")
        .replace("↑", "^")
    )
    return re.sub(r"[^\x20-\x7E\r\n]", "", text)

def build_llm_gpt_rub() -> ChatOpenAI:
    rub_token = os.getenv("RUBGPT_TOKEN")
    if not rub_token:
        raise RuntimeError(
            "RUBGPT_TOKEN doesn't exist"
        )
    model_name = os.getenv("LLM_MODEL")
    if not model_name:
        raise RuntimeError("LLM_MODEL is not set")

    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=rub_token, 
        base_url="https://gpt.ruhr-uni-bochum.de/external/v1", 
        timeout=600,
    )

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a data-extraction assistant.\n"
        "Use ONLY the context below (Global Hunger Index 2006–2008 PDFs).\n"
        "Extract every numeric indicator you can find. "
        "Return a CSV with columns Country, Indicator, Value, Source.\n"
        "If no numbers are present, answer exactly: No numeric data found\n"
        "=== context ===\n{context}\n=== end ===\n"
        "Question: {question}\nAnswer:"
    )
)

def ask(question_words: List[str], top_k: int) -> None:
    cfg_path = STORE_DIR / "params.json"
    cfg = json.loads(cfg_path.read_text())
    embedder = SentenceTransformerEmbeddings(model_name=cfg["emb_model"])
    db = FAISS.load_local(
         str(STORE_DIR),
         embedder,
         allow_dangerous_deserialization=True 
     )

    # llm = ChatOpenAI(model_name=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)

    q = " ".join(question_words)
    for i, d in enumerate(db.similarity_search(q, k=min(6, top_k)), 1):
        print(f"\n--- PREVIEW {i} ---\n{d.page_content[:300]}")

    llm = build_llm_gpt_rub()

    qa = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=db.as_retriever(
             search_kwargs={
                 "k": top_k,        
                 "score_threshold": 0.2      
             }
         ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    #result = qa(q)
    result = qa.invoke({"query": q}) 
    answer  = result["result"]
    sources = result["source_documents"]

    print("\nAnswer:\n", answer, "\n\nSources:")
    for doc in sources:
        print(" -", doc.metadata)

    # Save PDF-file
    pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, f"Question:\n{sanitize(q)}\n")
    pdf.multi_cell(0, 6, f"Answer:\n{sanitize(answer)}\n")
    pdf.multi_cell(0, 6, "Sources:")
    for doc in sources:
        meta = "; ".join(f"{k}={v}" for k, v in doc.metadata.items())
        pdf.multi_cell(0, 6, sanitize(meta))
    fname = f"answer_{datetime.now():%Y%m%d_%H%M%S}.pdf"
    pdf.output(str(STORE_DIR.parent / fname))
    print("Saved →", fname)

def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG index")
    parser.add_argument("question", nargs="*", help="Free-form question")
    parser.add_argument("-k", type=int, default=DEFAULT_TOP_K, help="How many chunks to retrieve")
    parser.add_argument("--file", "-f", type=str, help="Path to a .txt file with a prompt")
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            ask([fh.read()], args.k)
    else:
        if not args.question:
            parser.error("Provide a question or use --file")
        ask(args.question, args.k)


if __name__ == "__main__":
    main()
    