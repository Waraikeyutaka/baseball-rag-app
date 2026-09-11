

# baseball-rag-app/ingest.py

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

#初期設定と API クライアント作成

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1") 
EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small") 
 
if not API_KEY: 
    print("Error: OPENROUTER_API_KEY が .env に設定されていません。") 
    sys.exit(1) 
 
client = OpenAI(api_key=API_KEY, base_url=BASE_URL) 
 
 
def embed(texts): 
    """OpenRouter の /embeddings エンドポイントを呼び出す""" 
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts) 
    return [d.embedding for d in resp.data] 
 
 
def split_into_chunks(text, chunk_size=400, overlap=80): 
    """固定長ベースのチャンク分割""" 
    chunks = [] 
    start = 0 
    n = len(text) 
    while start < n: 
        end = min(start + chunk_size, n) 
        chunks.append(text[start:end]) 
        if end == n: 
            break 
        start = end - overlap 
    return chunks 
 
 
# Chroma 永続化クライアント 
persist_dir = "./chroma_db" 
chroma = chromadb.PersistentClient(path=persist_dir) 
 
COLL_NAME = "baseball_players" 
try: 
    chroma.delete_collection(COLL_NAME) 
except Exception: 
    pass 
collection = chroma.create_collection(COLL_NAME) 
 
# data/ 配下の Markdown を読み込んで登録 
data_dir = Path("./data") 

#フォルダーの中をさらに掘って探す
md_files = sorted(
    list((data_dir / "hanshin").rglob("*.md")) +
    list((data_dir / "hiroshima").rglob("*.md"))
)



if not md_files: 
    print(f"Error: {data_dir} に .md ファイルが見つかりません。") 
    sys.exit(1) 
 
all_ids, all_docs, all_meta = [], [], [] 

for md in md_files:

    raw = md.read_text(encoding="utf-8")

    # 球団名
    team = md.parent.name

    # 選手名
    player = md.stem

    # チャンク分割
    chunks = split_into_chunks(raw)

    for i, c in enumerate(chunks):

        document_id = f"{team}_{player}_{i}"

        all_ids.append(document_id)
        all_docs.append(c)

        # メタデータに球団名、選手名、チャンク番号を追加する
        all_meta.append({
            "source": md.name,
            "player": player,
            "team": team,
            "chunk_index": i
        })

print(f"{len(all_docs)} 個のチャンクを埋め込み中...") 
embeddings = embed(all_docs) 
collection.add( 
    ids=all_ids, 
    documents=all_docs, 
    embeddings=embeddings, 
    metadatas=all_meta, 
) 
print(f"Chroma に {len(all_docs)} 件のチャンクを登録しました（{persist_dir}）。") 