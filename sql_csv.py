import os
import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def csv_to_sqlite(csv_path: str, db_path: str):
    """
    CSV 파일을 SQLite로 저장하는 함수 입니다.
    """
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    conn = sqlite3.connect(db_path)
    df.to_sql("accidents", conn, if_exists="replace", index=False)
    conn.close()
    print(f"✅ SQLite 저장 완료: {db_path}")


def load_filtered_rows(db_path: str, filters: dict) -> pd.DataFrame:
    """
    저장되어 있는 SQLITE에서 필터 조건에 맞는 데이터를 가져오는 함수 입니다.
    """
    where_clause = " AND ".join([f'"{k}" = "{v}"' for k, v in filters.items()])
    query = f"SELECT * FROM accidents WHERE {where_clause}"

    conn = sqlite3.connect(db_path)
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def rerank_by_cause_similarity(
    df: pd.DataFrame,
    user_cause: str,
    top_k: int = 5,
    model_name: str = "intfloat/multilingual-e5-large",
):
    """
    입력한 문장과 필터링된 SQLite 결과의 사고 원인과의 유사도를 비교하는 함수 입니다.
    """
    if df.empty:
        return pd.DataFrame()

    model = HuggingFaceEmbeddings(model_name=model_name)
    query_vec = model.embed_query(user_cause)
    query_vec = np.array([query_vec])
    cause_vecs = model.embed_documents(df["사고원인"].tolist())

    sims = cosine_similarity(query_vec, cause_vecs)[0]
    df = df.copy()
    df["유사도"] = sims
    top_cases = df.sort_values(by="유사도", ascending=False).head(top_k)

    return top_cases


def print_top_cases(top_cases: pd.DataFrame):
    if top_cases.empty:
        print("❌ 조건에 맞는 사고 사례가 없습니다.")
    else:
        for idx, row in top_cases.iterrows():
            print("\n🧠 유사도:", round(row["유사도"], 3))
            print("사고원인:", row["사고원인"])
            print("재발방지대책 및 향후조치계획:", row["재발방지대책 및 향후조치계획"])
