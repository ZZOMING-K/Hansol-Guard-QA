from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from kiwipiepy import Kiwi

def get_retriever(markdb, qadb, k, docs, dataset_docs):

    # kiwi tokenizer for bm25
    kiwi = Kiwi()

    # kiwi 토크나이징 함수(키워드 기반)
    def preprocessing_with_kiwi(text):
        return [t.form for t in kiwi.tokenize(text)]

    # pdf(markdown) retriever
    pdf_retriever = markdb.as_retriever(search_kwargs={'k': k}, search_type='similarity')

    # qa retriever
    qa_retriever = qadb.as_retriever(search_kwargs={'k': k}, search_type='similarity')

    # BM25 Retirever
    bm_pdf_retriever = BM25Retriever.from_documents(docs,
                                                    preprocess_func=preprocessing_with_kiwi)
    bm_qa_retriever = BM25Retriever.from_documents(dataset_docs,
                                                   preprocess_func=preprocessing_with_kiwi)
    bm_pdf_retriever.k = k
    bm_qa_retriever.k = k

    # ensemble retriever
    ensemble_pdf_retriever = EnsembleRetriever(
        retrievers=[pdf_retriever, bm_pdf_retriever],
        weights=[0.5, 0.5]
    )

    ensemble_qa_retriever = EnsembleRetriever(
        retrievers=[qa_retriever, bm_qa_retriever],
        weights=[0.5, 0.5]
    )

    # Reranker 사용
    encoder_model = HuggingFaceCrossEncoder(model_name='BAAI/bge-reranker-v2-m3')
    compressor = CrossEncoderReranker(model=encoder_model, top_n=3)

    final_doc_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                         base_retriever=ensemble_pdf_retriever)
    final_qa_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                        base_retriever=ensemble_qa_retriever)

    return final_doc_retriever, final_qa_retriever