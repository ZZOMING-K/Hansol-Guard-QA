import os 
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langchain_core.prompts import ChatPromptTemplate ,  MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from preprocessing import prepro_data
from retriever import get_retriever
from embedding import load_embedding_model , load_vector_db , load_pdf_docs , load_csv_docs
from generate import load_llm , generate_response

# data load 
data = prepro_data(path = './data/prepro_data.csv') 

# pdf_docs , csv_docs 
csv_dataset = load_csv_docs(data = data)
pdf_dataset = load_pdf_docs(pdf_path = './pdf2txt/')

# embedding model
hf_embeddings = load_embedding_model(model_name = "intfloat/multilingual-e5-large") 

# vector db 
pdf_db , qa_db = load_vector_db(hf_embeddings) 

# retriever  

pdf_retriever , csv_retriever = get_retriever(pdf_db , qa_db , 5 , pdf_dataset , csv_dataset) 

#load llm 
llm_model , tokenizer = load_llm(model_id = "Qwen/QwQ-32B")

class GraphState(TypedDict) :
    question : List[str]
    csv_prompt : str
    pdf_prompt : str
    pdf_docs : List[str]
    csv_docs : List[str]
    think_response : str
    final_response : str
    generator : object
    
def transfomr_query(state) : 
    
    print("--TRANSFORM QUERY--")
    
    question = state['question']
    
    work_clf  = question[0] # 공종 
    work_process = question[1] # 작업프로세스 
    accident_object = question[2] # 사고객체
    human_accident = question[3] # 인적사고
    property_accident = question[4] # 물적사고
    prompt = question[5] # 사고원인 
    
    csv_prompt = f"""{work_clf} 중 {prompt} 로 인해 사고가 발생했습니다.
    해당 사고는 {work_process} 중 발생했으며, 관련 사고 객체는 {accident_object} 입니다.
    이로 인한 인적피해는 {human_accident} 이고, 물적피해는 {property_accident} 로 확인됩니다."""
        
    pdf_prompt =  f"""{work_clf}({accident_object}) 관련 {work_process} 중 {prompt}으로 인해 발생한 
    인적사고 : {human_accident} 및 물적사고 : {property_accident} 에 대한 안전 작업 지침 및 안전 조치 사항"""
    
    return { "pdf_prompt" : pdf_prompt , "csv_prompt" : csv_prompt }

    

def retrieve(state) :

    print("--RETRIEVE--")

    csv_prompt = state['csv_prompt']
    pdf_prompt = state['pdf_prompt'] 

    csv_docs = csv_retriever.invoke(csv_prompt)
    pdf_docs = pdf_retriever.invoke(pdf_prompt)
    
    example_prompt , related_pdf_prompt = [] , []

    
    for i , doc in enumerate(csv_docs) :
        
        context = doc.page_content
        question = doc.metadata['question']
        answer = doc.metadata['answer']
        example_prompt.append(f'유사 사고 사례 {i} : {context} {question}\n\n대응책:{answer}\n\n')
        
    for i , doc in enumerate(pdf_docs) : 
        
        context = doc.page_content
        source = doc.metadata['source'].split('/')[-1].split('md')[0]
        
        related_pdf_prompt.append(f'안전 지침서 {i} : {source}\n\n{context}\n')

    return {"pdf_docs" : related_pdf_prompt, "csv_docs" : example_prompt, "csv_prompt" : csv_prompt}


def generate(state) :

    print("--GENERATE--")

    question = state['csv_prompt']
    pdf_docs = state['pdf_docs']
    csv_docs = state['csv_docs']
    
    stream_generator = generate_response(llm_model, tokenizer, pdf_docs, csv_docs, question)

    return {"generator" : stream_generator , "think_response" : "" , "final_response" :  ""}

workflow = StateGraph(GraphState)

workflow.add_node("transform_query", transform_query) 
workflow.add_node("retrieve", retrieve)  
workflow.add_node("generate", generate)  

# Build Graph
workflow.add_edge(START, "transform_query") 
workflow.add_edge("transform_query", "retrieve") 
workflow.add_edge("retrieve" , "generate") 
workflow.add_edge("generate" , END)

# Compile
graph = workflow.compile()
