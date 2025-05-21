#!/usr/bin/env python3
import pickle
from pathlib import Path
from typing import List, Dict

import pdfplumber
import camelot
import tabula
import pandas as pd
from slugify import slugify
from tqdm import tqdm

PDF_DIR = Path("data") # PDF here
CACHE_FILE = Path("interim/ingested.pkl") 

CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

def clean_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = [slugify(col) for col in df.columns]
    df = df[~df.apply(lambda r: (r == df.columns).all(), axis=1)]
    df = df.apply(pd.to_numeric, errors="ignore")
    df = df.loc[:, (df != df.iloc[0]).any()]
    return df.reset_index(drop=True)

def extract_tables_with_fallback(pdf_path: Path) -> List[pd.DataFrame]:
    try:
        return [t.df for t in camelot.read_pdf(str(pdf_path), pages="all", flavor="stream")]
    except Exception:
        return tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)

def main() -> None:
    corpus: List[Dict] = []

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                corpus.append(
                    {
                        "id": f"{pdf_path.stem}_p{page_num}",
                        "kind": "page",
                        "content": page.extract_text() or "",
                        "meta": {"file": pdf_path.name, "page": page_num},
                    }
                )

        for tbl_num, raw_df in enumerate(extract_tables_with_fallback(pdf_path), start=1):
            corpus.append(
                {
                    "id": f"{pdf_path.stem}_t{tbl_num}",
                    "kind": "table",
                    "content": clean_table(raw_df).to_csv(index=False),
                    "meta": {"file": pdf_path.name, "table": tbl_num},
                }
            )

    print(f"Extracted {len(corpus)} items from {len(pdf_files)} PDF(s)")
    with CACHE_FILE.open("wb") as fh:
        pickle.dump(corpus, fh)

if __name__ == "__main__":
    main()
