from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader, DataFrameLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import KonlpyTextSplitter

def langchain_embeddings(embedding_name):
    '''
    랭체인 임베딩 설정 함수.
    :param config:
    :return:
    '''
    embedding = HuggingFaceEmbeddings(embedding_name,
                                      model_kwargs = {'device' : 'cuda'},
                                      encode_kwargs={'normalize_embeddings': True})

    return embedding

def document_loader():
    '''
    문서 업로드 함수
    :param config:
    :return:
    '''
    # 마크다운 처리
    mark_path = './pdf2txt'
    text_loader_kwargs = {"autodetect_encoding": True}

    # markdown loader
    loader = DirectoryLoader(
        mark_path,
        glob="*.md",
        loader_cls=TextLoader,
        silent_errors=True,
        loader_kwargs=text_loader_kwargs,
    )
    docs = loader.load()

    # 잘 올라갔는지 확인

    print(docs[0].page_content)
    print(len(docs))

    # 마크다운 문서 처리
    # 헤더로 나누어져있지 않아 기존 코드 사용
    text_splitter = KonlpyTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(docs)

    loader = DataFrameLoader('./data/train.csv', page_content_column="context")
    datasets_docs = loader.load()

    return docs, datasets_docs


def store_vectordb(docs, save_path):
    # FAISS가 빨라서 사용했습니다.
    embeddings = langchain_embeddings('intfloat/multilingual-e5-large')
    vectordb = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)
    vectordb.save_local(save_path)

    return vectordb
