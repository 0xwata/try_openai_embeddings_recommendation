import openai
from openai.embeddings_utils import cosine_similarity
import os
import pandas as pd
import json

openai.api_key = [OPEN_AI_KEY]

QUERY = "プロジェクトマネジメント 英語 コミュニケーション能力 リーダーシップ 協調性 責任感"
# 検索用の文字列をベクトル化
query = openai.Embedding.create(model="text-embedding-ada-002", input=QUERY)
query = query["data"][0]["embedding"]


# データベースの読み込み
with open("data/ideal_candidate_profile_index.json") as f:
    INDEX = json.load(f)

# 総当りで類似度を計算
results = map(
    lambda i: {
        "strings": i["strings"],
        # ここでクエリと各文章のコサイン類似度を計算
        "similarity": cosine_similarity(i["embedding"], query),
    },
    INDEX,
)
# コサイン類似度で降順（大きい順）にソート
results = sorted(results, key=lambda i: i["similarity"], reverse=True)


# 以下で結果を表示
print(f"Query: {QUERY}")
print("Rank: Strings Similarity")
for i, result in enumerate(results):
    print(f'{i+1}: {result["strings"]} {result["similarity"]}')

print("====Best Doc====")
print(f'strings: {results[0]["strings"]}')

print("====Worst Doc====")
print(f'strings: {results[-1]["strings"]}')
