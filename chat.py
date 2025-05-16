from typing import TypedDict, List, Optional
import time
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from torch.jit import strict_fusion

from Logger import Logger
from langchain_ollama.llms import OllamaLLM
from chain import store_vectordb, document_loader
from embedding import load_csv_docs, load_pdf_docs, load_embedding_model, load_vector_db
from generate import generate_response
from preprocessing import prepro_data
from retriever import get_retriever
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, RemoveMessage, HumanMessage
import redis
import json


# data load
data = prepro_data(path='./data/prepro_data.csv')

# pdf_docs , csv_docs
csv_dataset = load_csv_docs(data=data)
pdf_dataset = load_pdf_docs(pdf_path='./pdf2txt/')

# vector db
pdf_db, qa_db = load_vector_db()

pdf_retriever, csv_retriever = get_retriever(pdf_db, qa_db, k=30, pdf_dataset, csv_dataset)

# global llm
llm = OllamaLLM(model='exaone-deep:7.8b')

# redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)


class ChatHistoryEntry(BaseModel):
    id: str
    question: str
    pdf_question: str
    generation: str
    pdf_docs: List[str]
    csv_docs: List[str]
    summary: str

class ChatGraphState(BaseModel) :
    user_id: str # user_id를 숫자로 하면 망각 및 삭제 시 관리가 힘들어지는 단점 존재.
    question : str
    pdf_question : str
    generation : str
    pdf_docs : List[str]
    csv_docs : List[str]
    summary: str
    chat_history: List[ChatHistoryEntry]
    forget_command: int


def summarize_conversation(state: ChatHistoryEntry):
    # 요약이 있으면, 시스템 메시지로 추가
    summary = state.get('summary', '')
    if summary:
        # 이미 요약이 있는 경우, 기존의 메시지 번환
        return {'messages': state['messages']}
    else:
        summary_message = 'Create a summary of the conversation above in Korean'

        messages = state['messages'] + [HumanMessage(content=summary_message)]

        # 요약이 중점이기 때문에 rag 대신 llm만 호출
        response = llm.invoke(messages)

        return {'messages': [response]}


def chat_memory(state: ChatGraphState):
    keys = redis_client.keys(f"chat:{state['user_id']}:*")
    state['chat_history'] = []

    # key 여부 조회
    for key in keys:
        data = redis_client.get(key)
        if data:
            state['chat_history'].append(ChatHistoryEntry(**json.loads(data)))

    # forget하지 않는다면, 기억







def chat_run():

    log = Logger()
    start_time = time.time()
    log.info(f'시작 시간 : {start_time}')

    workflow = StateGraph(ChatGraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_pdf_query)
    workflow.add_node('summarize', summarize_conversation)

    # Build Graph
    workflow.add_edge(START, "transform_query")
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", END)

    # Compile
    graph = workflow.compile()

    end_time = time.time()
    log.info(f'종료 시간 : {end_time}')
    log.info(f'총 걸린 시간 : {end_time - start_time}')


def transform_pdf_query(state):
    system = """
        당신은 건설 안전 분야 전문 쿼리 변환기입니다. 입력된 건설 사고 관련 질문을 안전보건작업지침 문서 벡터 검색에 최적화된 형태로 재작성하는 역할을 합니다.
        """

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    question = state['question']

    better_question = question_rewriter.invoke({"question": question})

    print(better_question)

    return {"pdf_question": better_question}


def retrieve(state: ChatGraphState):
    print("--RETRIEVE--")

    csv_question = state['question']  # origin question
    pdf_question = state['pdf_question']  # 재작성된 쿼리

    csv_docs = csv_retriever.invoke(csv_question)
    pdf_docs = pdf_retriever.invoke(pdf_question)

    example_prompt = []

    for i, doc in enumerate(csv_docs):
        context = doc.page_content
        question = doc.metadata['question']
        answer = doc.metadata['answer']

        example_prompt.append(f'question:{context} {question}\nanswer:{answer}\n\n')


    return {"pdf_docs": pdf_docs, "csv_docs": example_prompt, "question": csv_question, "pdf_question": pdf_question}


def generate(state):
    print("--GENERATE--")

    question = state['question']

    pdf_docs = state['pdf_docs']
    csv_docs = state['csv_docs']

    rag_chain = generate_response(pdf_docs, csv_docs, question)

    generation = rag_chain.invoke({"pdf_docs": pdf_docs, "csv_docs": csv_docs, "question": question})

    return {"pdf_docs": pdf_docs, "csv_docs": csv_docs, "question": question, "generation": generation}



def grade_documents(state):
    class GradeDocuments(BaseModel):
        """Binary score for relevance check on retrieved documents."""

        binary_score: str = Field(
            description="Documents are relevant to the question, 'yes' or 'no'"
        )

    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {pdf_docs} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader

    print("---CHECK PDF DOCUMENT RELEVANCE TO QUESTION---")

    pdf_question = state["pdf_question"]

    pdf_docs = state["pdf_docs"]

    filtered_pdf_docs = []

    for p_doc in pdf_docs:

        score = retrieval_grader.invoke({"question": pdf_question, "pdf_docs": p_doc.page_content})

        grade = score.binary_score

        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_pdf_docs.append(p_doc)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    if not filtered_pdf_docs:
        print("참조할 안전보건지침서가 없습니다.")

    return {"pdf_docs": filtered_pdf_docs, "pdf_question": pdf_question}

if __name__ == '__main__':
    chat_run()
