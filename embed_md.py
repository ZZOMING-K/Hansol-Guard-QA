from typing import List, Tuple
import os
import glob
import numpy as np
import re
import pickle
import faiss
from langchain.docstore.document import Document
from langchain_experimental.text_splitter import SemanticChunker

# from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_markdown_files(folder_path: str) -> List[Tuple[str, str]]:
    """
    마크 다운 파일들을 로드하여(파일명, 텍스트) 리스트로 반환하는 함수
    """

    file_paths = glob.glob(os.path.join(folder_path, "*.md"))

    return [(path, open(path, "r", encoding="utf-8").read()) for path in file_paths]


def custom_md_splitter(md_text: str) -> List[Document]:
    """
    마크다운 헤더가 현재 md 파일에 없기 때문에 챕터 구분 별로 (예 : 1. 목적 2. 배경)
    나누기 위한 함수.
    """
    pattern = re.compile(r"(?P<header>^\d+(\.\d+)*\s+.+)", re.MULTILINE)
    matches = list(pattern.finditer(md_text))

    documents = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)

        header = match.group("header").strip()
        content = md_text[start:end].strip()

        # 컨텐츠에는 헤더로 분리한 것도 내용과 합치고, 메타데이터에서 "section" 으로 구분
        if content:
            doc = Document(
                page_content=f"{header}\n{content}", metadata={"section": header}
            )
            documents.append(doc)

    return documents


def semantic_chunk_documents(
    documents: List[Document], model: HuggingFaceEmbeddings
) -> List[str]:
    """
    텍스트를 의미적 유사도 기준으로 청킹하는 함수.
    """
    chunker = SemanticChunker(embeddings=model)
    final_chunks = []
    for doc in documents:
        chunks = chunker.split_text(doc.page_content)
        final_chunks.extend(chunks)
    return final_chunks


def load_existing_faiss(output_dir: str):
    """
    기존 저장된 벡터DB 가져오는 함수
    """
    faiss_path = os.path.join(output_dir, "faiss.index")
    meta_path = os.path.join(output_dir, "metadata.pkl")

    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        # 기존 파일 경로 추출
        existing_paths = set([line.split(" | ")[0] for line in metadata])
        return index, metadata, existing_paths
    else:
        return None, [], set()


def save_faiss(index, metadata: List[str], output_dir: str):
    """
    벡터DB 저장 함수
    """
    os.makedirs(output_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(output_dir, "faiss.index"))
    with open(os.path.join(output_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    print(f"✅ 저장 완료: {output_dir}")


def process_md_folder(
    md_folder_path: str,
    output_dir: str,
    model_name="intfloat/multilingual-e5-large",
):
    """
    최종적으로 벡터DB를 만드는 함수. 입력한 경로에 기존 벡터DB가 없다면
    새로 생성한다.

    # 파라메터
    - md_folder_path : md 파일이 모여있는 폴더 이름 입력
    - output_dir : 기존 저장되어 잇는 벡터 (md : vectordb/faiss_md_index)
    """
    model = HuggingFaceEmbeddings(model_name=model_name)
    md_files = load_markdown_files(md_folder_path)

    # 기존 인덱스, 메타데이터, 파일경로 목록 로드
    index, metadata, existing_paths = load_existing_faiss(output_dir)
    if index is None:
        print("🔄 새로운 FAISS 인덱스 생성 중...")
        sample_vec = model.embed_documents(["임시 문장"])[0]
        dim = len(sample_vec)
        index = faiss.IndexFlatL2(dim)
        metadata = []
        existing_paths = set()

    new_chunks = []
    new_metadata = []

    for path, text in md_files:
        if path in existing_paths:
            print(f"⏭ 이미 처리됨: {path}")
            continue

        docs = custom_md_splitter(text)
        chunks = semantic_chunk_documents(docs, model)
        print(f"📄 {os.path.basename(path)} → {len(chunks)} chunks")

        new_chunks.extend(chunks)
        new_metadata.extend([f"{path} | {chunk[:50]}" for chunk in chunks])

    if new_chunks:
        embeddings = model.embed_documents(new_chunks)
        embeddings = np.array(embeddings).astype("float32")
        index.add(embeddings)
        metadata.extend(new_metadata)
        save_faiss(index, metadata, output_dir)
    else:
        print("✅ 추가할 새로운 문서가 없습니다.")


def search_md_faiss(
    query: str,
    index_dir: str,
    top_k: int = 5,
    model_name: str = "intfloat/multilingual-e5-large",
) -> List[Tuple[str, float]]:
    """
    마크다운 벡터DB에 질의(query)를 검색하고 유사한 청크 반환

    Returns: List of (chunk text, score)
    """
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "metadata.pkl")

    # 파일 체크
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            "FAISS index or metadata.pkl not found in the directory."
        )

    # 로딩
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    # 질의 임베딩
    model = HuggingFaceEmbeddings(model_name=model_name)
    query_vector = model.embed_query(query)
    query_vector = np.array([query_vector]).astype("float32")

    # 검색
    distances, indices = index.search(query_vector, top_k)
    results = []
    for i, score in zip(indices[0], distances[0]):
        if i < len(metadata):
            results.append((metadata[i], score))

    return results
