from embed_md import process_md_folder, search_md_faiss
from sql_csv import (
    csv_to_sqlite,
    load_filtered_rows,
    rerank_by_cause_similarity,
    print_top_cases,
)

if __name__ == "__main__":
    out_path = "vectordb/faiss_md_index"
    md_path = "pdf2txt"

    ## 벡터DB  테스트 부분
    # 벡터 DB 추가할 때 사용하는 코드 입니다.
    # process_md_folder(md_folder_path=md_path, output_dir=out_path)

    # 문장 쿼리로 검색하는 부분입니다.
    # y = search_md_faiss(
    #     "콘크리트 펌프카를 이용하여 콘크리트를 타설 시", "vectordb/faiss_md_index"
    # )
    # print(y)

    ## SQLite 테스트 부분

    # SQLite 저장할 경로 입니다.
    db_path = "db/accident.sqlite"
    csv_path = "data/prepro_train.csv"

    # SQL을 저장할 때 최초 1회만 실행 합니다.
    # csv_to_sqlite(csv_path=csv_path, db_path=db_path)

    # 필터링 조건 입니다.
    filters = {
        "인적사고": "끼임",
        "물적사고": "없음",
        "작업프로세스": "설치작업",
        "사고객체": "건설자재 > 자재",
    }
    # 입력 사고 원인
    user_cause = "관리 직원의 불안전 행동으로 자재를 발판 사용으로 인하여 주변 근로자(판넬설치공) 발등에 떨어짐"

    # 같은 조건의 필터링 데이터를 가져옵니다.
    filtered_df = load_filtered_rows(db_path=db_path, filters=filters)

    # 입력 사고 원인과 필터링 데이터와의 유사도를 계산합니다.
    top_cases = rerank_by_cause_similarity(df=filtered_df, user_cause=user_cause)

    # 유사도와 결과 출력
    print_top_cases(top_cases=top_cases)
