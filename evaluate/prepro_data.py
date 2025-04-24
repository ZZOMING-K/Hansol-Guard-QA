import sys
import os
import pandas as pd
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from preprocessing import prepro_data
from embedding import load_csv_docs, load_pdf_docs, load_embedding_model, load_vector_db
from retriever import get_retriever

def generate_prompt(row, csv_retriever, pdf_retriever):

    query = row['context']

    csv_docs = csv_retriever.invoke(query)
    pdf_docs = pdf_retriever.invoke(query)

    relate_example_prompt = []
    relate_pdf_prompt = []

    for i, doc in enumerate(csv_docs):
        context = doc.page_content
        question = doc.metadata.get('question', '')
        answer = doc.metadata.get('answer', '')
        relate_example_prompt.append(f'사고 상황: {context} {question}\n대응책: {answer}\n\n')

    for i, doc in enumerate(pdf_docs):
        context = doc.page_content
        source = doc.metadata.get('source', '알 수 없음').split('/')[-1].split('md')[0]
        relate_pdf_prompt.append(f'관련 안전 지침서 {i + 1}.\n\n{context}\n출처: {source}\n\n')

    return pd.Series({
        'related_example_prompt': ''.join(relate_example_prompt),
        'related_pdf_prompt': ''.join(relate_pdf_prompt)
    })

def generate_prompt_df(documents, csv_retriever, pdf_retriever):

    df = pd.DataFrame([{
        'context': doc.page_content,
        'question': doc.metadata.get('question', '')
    } for doc in documents])
    
    prompt_df = df.apply(lambda row: generate_prompt(row, csv_retriever, pdf_retriever), axis=1)
    df = pd.concat([df, prompt_df], axis=1)
    
    return df

def main():

    data = prepro_data(path='./data/prepro_data.csv')  # 학습 데이터
    test = prepro_data(path='./data/test.csv', drop_col=True)  # 테스트 데이터

    combined_test_data = load_csv_docs(test, answer_col=False)

    csv_dataset = load_csv_docs(data=data)
    pdf_dataset = load_pdf_docs(pdf_path='./pdf2txt/')

    hf_embeddings = load_embedding_model(model_name="intfloat/multilingual-e5-large")

    pdf_db, qa_db = load_vector_db(hf_embeddings)

    pdf_retriever, csv_retriever = get_retriever(pdf_db, qa_db, 4, pdf_dataset, csv_dataset)

    test_data = generate_prompt_df(combined_test_data, csv_retriever, pdf_retriever)
    print(test_data.head())

    test_data.to_csv('../data/test_data.csv', index=False)

if __name__ == "__main__":
    main()