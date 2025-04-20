import os
import glob
import re
import faiss
from langchain.docstore.document import Document
from langchain.document_loaders import DataFrameLoader

def split_documents(chunk_size, KB):

  text_splitter = RecursiveCharacterTextSplitter(
        separators=[ "\n\n", "\n", ".", " ", ""] ,
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 5),
        length_function = len ,
        is_separator_regex=True
    )

  # 문서 리스트를 순회하면서 청크 생성

  docs_processed = []

  for doc in KB:
      docs_processed += text_splitter.split_documents([doc])

  return docs_processed

def load_pdf_docs(pdf_path = './pdf2txt/') : 
    

    text_loader_kwargs = {"autodetect_encoding": True}

    loader = DirectoryLoader(
        pdf_path,
        glob="*.md",
        loader_cls=TextLoader,
        silent_errors=True,
        loader_kwargs=text_loader_kwargs,
    )
    
    docs = loader.load()
    
    docs_processed= split_documents(chunk_size = 500, KB = docs)
    
    return docs_processed

def load_csv_docs(data) : 
    
    combined_csv_data = data.apply(
    
    lambda row: {
        "context": (
            f"'{row['공종']}' 중 {row['사고원인']}'로 인해 사고가 발생했습니다. "
            f"해당 사고는 '{row['작업프로세스']}' 중 발생했으며, 관련 사고객체는 '{row['부위']}'입니다. "
            f"이로 인한 인적피해는 '{row['인적사고']}' 이고, 물적피해는 '{row['물적사고']}'로 확인됩니다."
        ),
        "question" : f"해당 사고의 재발 방지 대책과 향후 조치 계획은 무엇인가요?",
        "answer": row["재발방지대책 및 향후조치계획"]
    },
    axis=1
    )
    
    loader = DataFrameLoader(combined_csv_data, page_content_column="context")
    datasets_docs = loader.load()
    
    return dataset_docs

#  embedding model load 
def load_embedding_model(model_name = "intfloat/multilingual-e5-large") :

    embedding_model_name = model_name

    hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name ,
                                        model_kwargs = {"device" : "cuda" , "trust_remote_code" : True} , # cuda, cpu
                                        encode_kwargs = {"normalize_embeddings" : True}) # set True for cosine similarity
    
    return hf_embeddings

# vector db load 
def load_vector_db(embedding_model) : 
    
    pdf_db = FAISS.load_local('./vectordb/pdf_faiss' , 
                              embedding_model,
                              allow_dangerous_deserialization = True
                              )

    csv_db = FAISS.load_local('./vectordb/csv_faiss' , 
                              embedding_model ,
                              allow_dangerous_deserialization = True 
                              )
    
    return pdf_db , csv_db
