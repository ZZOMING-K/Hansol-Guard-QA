import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preproces(path): 
    
    data = pd.read_csv(path)  # 1차 전처리 완료된 데이터 
    data = data.drop(['검사여부', '삭제여부', '비고'], axis=1)
    col_to_drop = ['ID', '발생일시', '사고인지 시간', '날씨', '기온', '습도', '공사종류', '연면적', '층 정보', '장소']
    data = data.drop(col_to_drop, axis=1)
    data = data.drop_duplicates().reset_index(drop=True)

    # 불필요한 사고원인 제거
    exclude = ['.', '-', '작업자 요양급여 결정', '알수없음', '해당없음', '해당없음.']
    data = data[~data['사고원인'].isin(exclude)].reset_index(drop=True)

    # 오타 수정 및 결측 제거
    data['사고원인'] = data['사고원인'].replace('작업자 부주위', '작업자 부주의')
    data = data.dropna().reset_index(drop=True)

    return data 


def fill_data(data , st_model  = "BAAI/bge-m3", save_path = './data/prepro_data.csv') : 
    
    sentences = data['사고원인'].tolist()

    model = SentenceTransformer(st_model) # model_load

    reason_vectors = model.encode(sentences)
    prevention_vectors = model.encode(data['재발방지대책 및 향후조치계획'].tolist())

    target_rows = data[data['재발방지대책 및 향후조치계획'].isin(['작업자 단순과실로 인한 재발 방지 대책 및 향후 조치 계획 없음.' ,
                                                     '재발 방지 대책과 향후 조치 계획의 부재.'])]

    for idx in target_rows.index:

        cause_vector = model.encode(data.at[idx, '사고원인']).reshape(1, -1)

        # 유사도 계산
        similarities = cosine_similarity(cause_vector, reason_vectors).flatten()

        # 자기 자신을 제외한 상위 4개 선택
        top_n_idx = np.argsort(similarities)[::-1][1:5]  # 가장 유사한 4개 선택
        print(data.iloc[top_n_idx, data.columns.get_loc('사고원인')])

        # 상위 재발 방지 대책 및 향후 조치 계획 가져오기
        similar_plans = data.iloc[top_n_idx]['재발방지대책 및 향후조치계획'].tolist()

        plan_embeddings = model.encode(similar_plans) # 임베딩
        mean_vector = np.mean(plan_embeddings, axis=0) # 평균 계산

        # 가장 평균적인 문장 찾기
        similarities = cosine_similarity([mean_vector], prevention_vectors)[0]

        most_average_idx = np.argmax(similarities)
        recommended_plan = data['재발방지대책 및 향후조치계획'].iloc[most_average_idx]

        print(recommended_plan)
        data.at[idx, '재발방지대책 및 향후조치계획'] = recommended_plan

    # 저장 
    data.to_csv(save_path) 
    
    return 


def prepro_data(path) : 
    
    path = './data/prepro_data.csv' # 전처리된 데이터 경로
    
    data = pd.read_csv(path , index_col = 0)

    data['부위'] = data['부위'].str.split('/').str[0].str.strip() # 부위 나누기

    data = data.drop_duplicates().reset_index(drop = True)

    return data 
