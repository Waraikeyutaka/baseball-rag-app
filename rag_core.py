import os
from typing import Optional

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1"
)

EMBED_MODEL = os.getenv(
    "OPENROUTER_EMBED_MODEL",
    "openai/text-embedding-3-small"
)

CHAT_MODEL = os.getenv(
    "OPENROUTER_CHAT_MODEL",
    "nvidia/nemotron-3-super-120b-a12b:free"
)

PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    "./chroma_db"
)

COLL_NAME = os.getenv(
    "CHROMA_COLLECTION",
    "baseball_players"
)

HISTORY_TURNS = int(
    os.getenv("HISTORY_TURNS", "4")
)

TOP_K = int(
    os.getenv("SEARCH_TOP_K", "3")
)

TEMPERATURE = float(
    os.getenv("TEMPERATURE", "0.0")
)

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

chroma = chromadb.PersistentClient(
    path=PERSIST_DIR
)

collection = chroma.get_collection(
    COLL_NAME
)


def get_all_players() -> list[str]:
    """ChromaDBに登録されている選手名を取得"""

    data = collection.get(
        include=["metadatas"]
    )

    players = set()

    for metadata in data["metadatas"]:
        if metadata and metadata.get("player"):
            players.add(metadata["player"])

    return sorted(players)


def detect_players(query: str) -> list[str]:
    """質問文に含まれている選手名を検出"""

    player_name_map = {
        "佐藤輝明": "satoteru",
        "森下翔太": "morishita",
        "近本光司": "chikamoto",
        "中野拓夢": "nakano",
        "坂本誠志郎": "sakamoto",
        "伊藤将司": "ito",
        "大竹耕太郎": "ohtake",
        "村上頌樹": "murakami",
        "才木浩人": "saiki",
        "小園海斗": "kozono",
        "ファビアン": "fabian",
        "森翔平": "mori",
        "森下暢仁": "morishitan",
        "大瀬良大地": "osera",
        "坂倉将吾": "sakakura",
        "中村奨成": "nakamurasho",
        "末包昇大": "suekane",
        "玉村昇悟": "tamamura",
        "床田寛樹": "tokoda",
    }

    detected = []

    for player_name, player_id in player_name_map.items():
        if player_name in query:
            detected.append(player_id)

    return detected


SYSTEM_MESSAGE = """\
あなたはユーザーの質問に回答するチャットボットです。

回答については、「Sources:」以下に記載されている内容に基づいて回答してください。
回答は簡潔にしてください。

「Sources:」に記載されている情報以外の回答はしないでください。

情報が複数ある場合は、「Sources:」のあとに
[Source1]、[Source2]、[Source3]のように記載されますので、
それらに基づいて回答してください。

また、ユーザーの質問に対して、
Sources: 以下に記載されている内容に基づいて
適切な回答ができない場合は、
「すみません。わかりません。」と回答してください。

回答の中に情報源の提示は含めないでください。
例えば、回答の中に「[Source1]」や「Sources:」という形で
情報源を示すことはしないでください。
"""


def search_docs(
    query: str,
    n_results: int = TOP_K
) -> list[dict]:
    """クエリを検索して関連する選手データを取得"""

    print("質問:", repr(query))

    # 質問をEmbedding化
    q_emb = client.embeddings.create(
        model=EMBED_MODEL,
        input=query
    ).data[0].embedding

    # 選手名を検出
    detected_players = detect_players(query)

    # チーム・ポジションを検出
    detected_team = None
    detected_position = None

    if "広島" in query:
        detected_team = "hiroshima"
    elif "阪神" in query:
        detected_team = "hanshin"

    if "野手" in query:
        detected_position = "野手"
    elif "投手" in query:
        detected_position = "投手"

    print("検出したチーム:", detected_team)
    print("検出したポジション:", detected_position)
    print("検出した選手:", detected_players)

    
    # チーム＋ポジションの一覧検索
    

    if detected_team and detected_position:

        print("チーム＋ポジション検索を実行")

        # チームで絞り込む
        results = collection.get(
            where={
                "team": detected_team
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

        hits = []

        for i, doc in enumerate(results["documents"]):

            # Markdown本文に
            # 「投手」「内野手」「外野手」などが
            # 含まれているか確認
            if detected_position in doc:

                hits.append({
                    "id": results["ids"][i],
                    "document": doc,
                    "metadata": results["metadatas"][i],
                    "distance": 0,
                })

        print(
            "チーム＋ポジション検索結果:",
            len(hits),
            "件"
        )

        return hits

  
    #選手名が指定されている場合
   
    if detected_players:

        hits = []

        for player_id in detected_players:

            results = collection.query(
                query_embeddings=[q_emb],
                n_results=n_results,
                where={
                    "player": player_id
                }
            )

            for i, doc in enumerate(
                results["documents"][0]
            ):

                hits.append({
                    "id": results["ids"][0][i],
                    "document": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })

        print(
            "選手名検索結果:",
            hits
        )

        return hits

    
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results
    )

    print(
        "通常の検索結果:",
        results
    )

    hits = []

    for i, doc in enumerate(
        results["documents"][0]
    ):

        hits.append({
            "id": results["ids"][0][i],
            "document": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return hits


def build_sources(
    hits: list[dict]
) -> str:
    """[SourceN]: ... 形式に整形"""

    lines = []

    for i, h in enumerate(
        hits,
        start=1
    ):

        src = h["metadata"].get(
            "source",
            h["id"]
        )

        lines.append(
            f"[Source{i}] ({src}): "
            f"{h['document']}"
        )

    return "\n".join(lines)


def answer(
    question: str,
    history: Optional[list[dict]] = None,
    model: Optional[str] = None,
) -> dict:
    """RAGのEnd-to-End実行"""

    history = history or []

    # Step 1: 検索
    hits = search_docs(question)

    # Step 2: 出典整形
    sources_text = build_sources(hits)

    # Step 3: LLMに質問＋Sourcesを渡す

    messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE
        }
    ]

    # 会話履歴
    for h in history[-HISTORY_TURNS:]:

        messages.append({
            "role": h["role"],
            "content": h["content"]
        })

    # 現在の質問＋Sources
    user_msg = (
        f"{question}\n\n"
        f"Sources:\n"
        f"{sources_text}"
    )

    messages.append({
        "role": "user",
        "content": user_msg
    })

    # 使用するモデル
    selected_model = model or CHAT_MODEL

    response = client.chat.completions.create(
        model=selected_model,
        messages=messages,
        temperature=TEMPERATURE,
    )

    print("OpenRouter response:")
    print(response)

    if (
        response.choices is None
        or len(response.choices) == 0
    ):

        answer_text = (
            "AIから回答を取得できませんでした。"
        )

    else:

        answer_text = (
            response.choices[0]
            .message.content
        )

        if answer_text is None:
            answer_text = (
                "AIから回答を取得できませんでした。"
            )

    return {
        "answer": answer_text.strip(),
        "sources": hits,
        "model": selected_model,
    }