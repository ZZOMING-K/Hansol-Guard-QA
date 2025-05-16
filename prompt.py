import pandas as pd
from datasets import Dataset

def chat_template(df, type):
  if type == 'train': # 학습 데이터일때
    combined_train_data = df.apply(
    lambda row: {
        "context": (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다."
        ),
        'question': '재발 방지 대책 및 향후 조치 계획은 무엇인가요?',
        "answer": row["재발방지대책 및 향후조치계획"]
    },
    axis=1
)
    combined_train_data = pd.DataFrame(combined_train_data)  # 학습 데이터

    return combined_train_data
  elif type == 'pdf': # pdf 학습 데이터
    combined_test_data = df.apply(
    lambda row: {
        "context": (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다."
        ),
        'question': '재발 방지 대책 및 향후 조치 계획은 무엇인가요?'
    },
    axis=1
)
    combined_pdf_data = pd.DataFrame(combined_test_data)  # pdf 학습 데이터
    return combined_pdf_data

  elif type == 'test': # 테스트 데이터일때
    combined_test_data = df.apply(
    lambda row: {
        "context": (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
        ),
        'question': '재발 방지 대책 및 향후 조치 계획은 무엇인가요?'
    },
    axis=1
)
    combined_test_data = pd.DataFrame(combined_test_data)  # 테스트  데이터
    return combined_test_data

# 예시 코드 생성
def generate_example(query , pdf_retriever , train_retriever) :

    relevant_docs = pdf_retriever.invoke(query)
    pdf_docs = train_retriever.invoke(query)

    example_prompt = []

    for i , doc in enumerate(relevant_docs) :

        context = doc.page_content
        question = doc.metadata['question']
        answer = doc.metadata['answer']

        example_prompt.append(f'question:{context} {question}\nanswer:{answer}\n\n')

    return example_prompt , pdf_docs

def generate_prompt(pdf_docs , example_prompt , query) :

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "당신은 건설 현장에서 발생한 안전사고를 분석하고 재발 방지 대책 및 향후 조치 계획을 제시하는 AI 도우미입니다.\n\n"
                "다음에 주어지는 '사고 관련 안전지침'과 '유사한 사고사례 및 대응책'을 참고하여 답변을 작성하세요.\n\n"

                "### 1. 사고 관련 안전지침\n"
                "아래는 건설 현장에서 사고를 방지하기 위한 안전 지침입니다.\n"
                "\n"
                f"{pdf_docs}\n"
                "\n\n"

                "### 2. 유사한 사고 사례 및 대응책\n"
                "아래는 유사한 사고 사례와 그에 대한 재발 방지 대책 및 향후 조치 계획입니다.\n"
                "\n"
                f"{''.join(example_prompt)}\n"
                "\n"
                "서론, 배경설명, 추가설명 없이 대응책을 핵심 내용만 요약해서 간략하게 작성하세요.\n\n"
                "답변 형식은 위의 '유사한 사고 사례 및 대응책' 과 동일하게 유지하세요.\n"
                "반드시 한국어로 답변하며, 주어진 사고 상황에 따른 재발 방지 대책 및 향후 조치 계획을 제시하세요."},]
        }
    ]

    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": f'question:{query} 해당 사고의 재발 방지 대책과 향후 조치 계획은 무엇인가요?\nanswer:'},]
        }
    )

    return messages

def generate_text(query) :
    # retriever
    example_prompt , pdf_docs = generate_example(query , pdf_retriever, qa_retriever)
    # prompt
    messages = generate_prompt(pdf_docs , example_prompt , query)
    return messages

def template_data_ready(df, save_path):
    dataset = Dataset.from_pandas(df)

    # 데이터셋 저장 : 만약을 방지하기 위함
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    return dataset