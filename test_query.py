"""ターミナルから RAG を試す CLI ツール

使い方:
    python test_query.py "近本光司の2025年の成績を教えて"
"""

import sys

from rag_core import answer


def main():
    if len(sys.argv) < 2:
        print('Usage: python test_query.py "<プロ野球関連の質問>"')
        sys.exit(1)

    question = sys.argv[1]

    result = answer(question, history=None)

    print("=" * 60)
    print(f"Q: {question}")
    print(f"Model: {result['model']}")
    print("-" * 60)

    print("A:")
    print(result["answer"])

    print("-" * 60)

    print("Sources:")
    for i, s in enumerate(result["sources"], start=1):
        print(f"  [{i}] {s['id']} (distance={s['distance']:.4f})")
        print(f"      player: {s['metadata'].get('player')}")
        print(f"      team: {s['metadata'].get('team')}")
        print(f"      document: {s['document'][:300]}")

    print("=" * 60)


if __name__ == "__main__":
    main()