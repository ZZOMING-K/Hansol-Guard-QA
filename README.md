# HANSOl-Guard-QA (한솔데코 건설 사고 대응 시스템) 👷🏼

**본 프로젝트는 RAG(Retrieval-Augmented Generation)를 활용하여 건설현장의 과거 사고 사례 및 관련 안전 지침서 기반 최적의 사고 대응책을 생성하는 것을 목표로 하는 시스템입니다.**

-----

**목차**

1. [프로젝트 개요](#1개요)
2. [개발 과정](#2개발과정)
3. [워크플로우](#3워크플로우)
4. [파일구조](#4파일구조)
5. [향후 발전 계획](#5향후발전계획)

---

## 1.개요 

건설 사고 발생 시, 사고 원인 분석과 함께 재발 방지 대책 및 향후 조치 계획서를 제출해야 합니다. 그러나 현장에서는 관련 지침의 종류가 다양하고 필요한 정보만을 선별해 참고하기가 어려워, 대부분 현장 근무자의 판단에 의존해 문서를 작성하는 실정입니다. 이에 따라, 건설 현장의 사고 대응 역량을 강화하고 안전 관리 프로세스를 보다 체계적으로 구축하기 위해, 사고 상황 및 원인을 바탕으로 대응 방안을 자동으로 생성하는 시스템을 개발하였습니다.

<br>

<p align="center">
<img src="https://github.com/user-attachments/assets/e5f7ef68-1bdc-42ee-9f0c-0b9e931d2194" width="800"/>
</p>

<br>

## 2.개발과정 

2.1) 데이터 소개

- 23,422 건의 사고 상황 데이터 
(발생일시, 사고인지 시간, 날씨, 기온, 습도, 사고원인, 재발방지대책 및 향후조치계획 등)
- 104개의 건설안전지침관련 PDF 파일

2.2) 데이터 전처리
- CSV : 오타 및 띄어쓰기 수정, 동어 반복 제거, 불필요한 기호 삭제 등 
- PDF : 오픈소스모델인 [olmOCR](https://huggingface.co/allenai/olmOCR-7B-0225-preview) 을 활용하여 텍스트 추출 

2.3) 유사문서 임베딩 및 검색

- 유사 사고 사례 및 안전 관련 지침서 임베딩 
  - 모델 : 한국어 임베딩 모델인 [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) 활용
  - 벡터 DB : FAISS를 사용하여 벡터 임베딩 저장 및 유사도 검색 수행

- 유사 문서 검색 쿼리 생성 
```python
# csv query
f"""{공종} 중 {사고원인} 로 인해 사고가 발생했습니다.
해당 사고는 {작업프로세스} 중 발생했으며, 관련 사고 객체는 {부위} 입니다. 
이로 인한 인적피해는 {인적사고} 이고, 물적피해는 {물적사고} 로 확인됩니다."""

# pdf query 
f"""{공종}({부위}) 관련 {작업프로세스} 중 {사고원인}으로 인해 발생한 인적사고 : {인적사고} 및 물적사고 : {물적사고} 에 대한 안전 작업 지침 및 안전 조치 사항"""
```
- 유사문서 검색 
  - 키워드 기반 검색기(Kiwi+BM25 Retriever) 와 문맥 기반 검색기(Similarity Search) 를 앙상블하여 검색 
  - [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) 모델을 활용해 검색된 문서들의 순위를 재조정 

2.4) 답변 추론 및 생성

- [QwQ-32B 모델](https://huggingface.co/Qwen/QwQ-32B) 활용 (Reasoning Model) 
- `<think>` 태그를 활용하여 체계적인 분석 및 결론 도출에 대한 투명한 사고 과정 확인
- 관련문서 상위 3개를 프롬프트에 포함하여 논리적으로 대책을 생성하도록 유도
```python
f"""[발생한 사고]
{question}
[유사 사고 사례 및 대응책]
{csv_docs}
[사고 관련 안전 지침서]
{pdf_docs}
위의 유사사고사례와 관련 안전 지침서를 참고하여, 발생한 사고에 대한 최적의 재발방지 대책 및 향후 조치 계획을 간결하게 작성해주세요.
반드시 '재발 방지 대응책:' 으로 시작하는 2줄 이내의 간결한 문장으로 답변하세요.
하나씩 차근차근 생각한 뒤, 최종 대응 대책을 도출하세요."""
```
<br>

## 3.워크플로우 
1. 사용자는 **사고정보(공종,작업프로세스,사고객체,인적사고,물적사고)와 사고원인**을 입력합니다.
2. 입력된 정보들을 바탕으로 관련 문서를 효과적으로 찾기 위한 **검색 쿼리를 자동 생성**합니다.  
3. 생성된 쿼리를 활용해 **BM25 기반 키워드 검색, FAISS 기반 임베딩 유사도 검색**을 실시해 관련 문서 추출 후 **Reranking 모델을 활용해 결과를 재정렬**합니다.
4. 검색된 문서를 **LLM에 전달하여, 사고 원인에 적합한 최적의 대응 방안을 생성**합니다.
5. 사용자에게 **사고 상황에 따른 최적의 대응 방안** 과 **해당 결과가 도출된 추론 및 사고 분석 과정**을 전달힙니다. 

<br>

## 4.파일구조 
```bash
Hansol-Guard-QA
 ├── data
 │ ├── final_train.csv  
 │ ├── prepro_data.csv  
 ├── pdf2txt
 │ ├── F.C.M 교량공사 안전보건작업 지침.md
 │ ├── I.L.M 교량공사 안전보건작업 지침.md
 │ ├── ...
 │ └── 흙막이공사(지하연속벽) 안전보건작업 지침.md
 ├── vectordb
 │ ├── csv_faiss
 │ │ ├── index.faiss
 │ │ └── index.pkl
 │ ├── pdf_faiss
 │ │ ├── index.faiss
 │ │ └── index.pkl
 ├── .gitignore
 ├── agent_graph.py #langgraph 구축
 ├── embedding.py  #data load 및 embedding 
 ├── generate.py #LLM 답변 생성 
 ├── logger.py
 ├── main.py #streamlit 실행 
 ├── pdf_load.py #pdf 문서 텍스트 추출 
 ├── preprocessing.py #csv 데이터 전처리 
 ├── README.md
 ├── requirements.txt
 └──  retriever.py #유사문서 검색 
```
<br>

## 5.향후발전계획 
- 모델 서빙 속도 향상 (vLLM, FAST API)

