# rag_pipeline/ingest.py
import re, sys, json, uuid, camelot, pdfplumber, tabula
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from slugify import slugify
import pickle

DATA_DIR = Path("data") # pdf files 
OUT_FILE = Path("interim/ingested.pkl") # pages and tables

OUT_FILE.parent.mkdir(exist_ok=True, parents=True)

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)     
    df = df.dropna(how="all").dropna(axis=1, how="all")                     
    df.columns = [slugify(c) for c in df.columns]                           
    df = df[~df.apply(lambda r: (r == df.columns).all(), axis=1)]           
    df = df.apply(pd.to_numeric, errors="ignore")                           
    df = df.loc[:, (df != df.iloc[0]).any()]                                
    return df.reset_index(drop=True)

ingested = []       

for pdf_path in tqdm(sorted(DATA_DIR.glob("*.pdf"))):
    with pdfplumber.open(pdf_path) as pdf:
        for p_idx, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            ingested.append({
                "id": f"{pdf_path.stem}_p{p_idx}",
                "type": "page",
                "content": text,
                "meta": {"file": pdf_path.name, "page": p_idx}
            })
    

