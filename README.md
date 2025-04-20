# HANSOl-Guard-QA (한솔데코 건설 사고 대응 시스템) 👷🏼

- **본 프로젝트는 RAG(Retrieval-Augmented Generation)을 기반으로 건설공사 사고 데이터를 활용하여 높은 품질의 사고 대응책을 생성하는 것을 목표로 합니다.**
 
<br>

## 개발과정 

### 주어진 데이터 

### Load & Split & Embedding 

- CSV Data
  - 오타 및 띄어쓰기 수정, 동어 반복 제거. 불필요한 기호 삭제, 특정 날짜/시간/장소명 삭제
  - 사고 원인이 아래 값 중 하나인 경우 해당 데이터 삭제
    - `-`, `.`, `작업자 요양 급여 결정`, `알 수 없음`, `해당 없음`, `결측치 (NaN)`
  - 재발방지 대책 및 향후 조치 계획이 없는 경우 (or 부재) 
    - 사고 원인을 기준으로 유사한 데이터를 찾아 재발방지 대책 및 향후 조치 계획을 가져옴
    - 유사한 대응책들의 `평균 벡터`를 계산하여 평균 벡터에 `가장 가까운 대응책`으로 변환

- PDF Data
  - 오픈소스모델인 **olmOCR**을 활용하여 텍스트 추출 ([olmOCR](https://huggingface.co/allenai/olmOCR-7B-0225-preview))
  - Docling, Markitdown, Marker-PDF, llamaparser와 출력 결과를 비교했을 때 **표가 변형없이 그대로 출력됨을 확인.**  
  - PDF 내용 중 사진으로만 이루어진 페이지, 계획서 작성 지침 등 재발 방지 대책 및 향후 조치 계획에 관련이 없다고 판단되는 페이지 삭제 진행.

### Retriever & Generation 

- Retriever

- ReRank

- Prompt

- Inference
  -  [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) 모델 활용
 

## 파일 구조 


## 향후 개선 방향 
