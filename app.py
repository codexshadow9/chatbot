import os, re, sys
from pathlib import Path

# ---- Optional: .env support ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from rank_bm25 import BM25Okapi

# --- Simple loaders (PDF, DOCX, TXT/MD) ---
def load_txt_md(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.read_text(encoding="latin-1", errors="ignore")

def load_pdf(path: Path) -> list[tuple[str, dict]]:
    text_pages = []
    try:
        from pypdf import PdfReader
    except Exception as e:
        raise RuntimeError("Install pypdf:  python -m pip install pypdf") from e
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            text_pages.append((txt, {"source": str(path), "page": i}))
    return text_pages

def load_docx(path: Path) -> str:
    try:
        import docx2txt
    except Exception as e:
        raise RuntimeError("Install docx2txt:  python -m pip install docx2txt") from e
    return docx2txt.process(str(path)) or ""

def iter_documents(data_dir="data"):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Folder '{data_dir}' not found. Create it and add PDFs/DOCX/TXT/MD.")
    for p in sorted(root.rglob("*")):
        if not p.is_file(): 
            continue
        suf = p.suffix.lower()
        if suf in {".txt", ".md"}:
            txt = load_txt_md(p)
            if txt.strip():
                yield (txt, {"source": str(p), "page": None})
        elif suf == ".pdf":
            for txt, meta in load_pdf(p):
                yield (txt, meta)
        elif suf == ".docx":
            txt = load_docx(p)
            if txt.strip():
                yield (txt, {"source": str(p), "page": None})

# --- Chunking & retrieval ---
def chunk_text(text: str, size=1000, overlap=150):
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += max(1, size - overlap)
    return chunks

def build_corpus(data_dir="data"):
    docs = list(iter_documents(data_dir))
    if not docs:
        raise ValueError("No supported files found in 'data'. Add PDF/DOCX/TXT/MD.")
    chunks, metas = [], []
    for txt, meta in docs:
        for ch in chunk_text(txt):
            chunks.append(ch)
            metas.append(meta)
    return chunks, metas

def tokenize(s: str):
    return re.findall(r"\w+", s.lower())

def build_bm25(corpus):
    tokenized = [tokenize(c) for c in corpus]
    return BM25Okapi(tokenized)

def format_context(chunks, metas, idxs):
    lines = []
    for i in idxs:
        m = metas[i]
        src = Path(m["source"]).name if m.get("source") else "?"
        page = m.get("page")
        tag = f"[source: {src}{', page '+str(page+1) if isinstance(page,int) else ''}]"
        lines.append(f"{tag}\n{chunks[i]}")
    return "\n\n".join(lines)

def pretty_sources(metas, idxs, k=4):
    out = []
    seen = set()
    for i in idxs:
        m = metas[i]
        src = Path(m.get("source","?")).name
        page = m.get("page")
        key = (src, page)
        if key in seen: 
            continue
        seen.add(key)
        page_info = f", p.{page+1}" if isinstance(page,int) else ""
        out.append(f"{src}{page_info}")
        if len(out) >= k: 
            break
    return "; ".join(out)

# --- LLM backends (OpenAI or Gemini) ---
def answer_with_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Install openai:  python -m pip install openai") from e
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You are a helpful assistant. Use ONLY the provided context. If the answer is not in the context, say you don't know."},
            {"role":"user","content":prompt},
        ],
        temperature=0,
    )
    return (resp.choices[0].message.content or "").strip()

def answer_with_gemini(prompt: str) -> str:
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("Install google-generativeai:  python -m pip install google-generativeai") from e
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    return (getattr(resp, "text", "") or "").strip()

def choose_provider():
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError("Set GOOGLE_API_KEY or OPENAI_API_KEY.")

def make_prompt(question: str, context: str) -> str:
    return (
        "Answer the question using ONLY the context below. "
        "If not answerable from context, say \"I don't know.\"\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

def main():
    provider = choose_provider()
    print(f"Using provider: {'Google Gemini' if provider=='gemini' else 'OpenAI'}")
    print("Building index from ./data ...")
    chunks, metas = build_corpus("data")
    bm25 = build_bm25(chunks)
    print(f"Indexed {len(chunks)} chunks. ✅")

    print("\nAsk a question about your documents. Type 'exit' to quit.\n")
    while True:
        q = input("Your question > ").strip()
        if not q:
            continue
        if q.lower() in {"exit","quit","q"}:
            print("Bye!")
            break

        scores = bm25.get_scores(tokenize(q))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:4]
        context = format_context(chunks, metas, top_idx)
        prompt = make_prompt(q, context)

        try:
            if provider == "openai":
                ans = answer_with_openai(prompt)
            else:
                ans = answer_with_gemini(prompt)
        except Exception as e:
            print(f"LLM error: {e}")
            continue

        print("\nAnswer:\n" + ans)
        print("\nSources: " + pretty_sources(metas, top_idx) + "\n")

if __name__ == "__main__":
    main()
