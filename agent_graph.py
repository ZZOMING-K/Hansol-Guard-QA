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
    question : str
    pdf_prompt : str
    generation : str
    pdf_docs : List[str]
    csv_docs : List[str]

def retrieve(state) :

    print("--RETRIEVE--")

    question = state['question']
    pdf_prompt = state['pdf_prompt'] 

    csv_docs = csv_retriever.invoke(question)
    pdf_docs = pdf_retriever.invoke(pdf_prompt)
    
    example_prompt = []
    
    for i , doc in enumerate(csv_docs) :
        
        context = doc.page_content
        question = doc.metadata['question']
        answer = doc.metadata['answer']
      
        example_prompt.append(f'question:{context} {question}\nanswer:{answer}\n\n')

    return {"pdf_docs" : pdf_docs, "csv_docs" : example_prompt, "question" : question, "pdf_prompt" : pdf_prompt}

def generate(state) :

    print("--GENERATE--")

    question = state['question']

    pdf_docs = state['pdf_docs']
    csv_docs = state['csv_docs']
    response = generate_response(llm_model, tokenizer, pdf_docs, csv_docs, question)
 
    generation = rag_chain.invoke({"pdf_docs": pdf_docs, "csv_docs" : csv_docs ,"question": quetion})

    return {"pdf_docs" : pdf_docs , "csv_docs" : csv_docs , "question" : question , "generation" : generation}

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)  
workflow.add_node("generate", generate)  

# Build Graph
workflow.add_edge(START, "retrieve") 
workflow.add_edge("retrieve" , "generate") 
workflow.add_edge("generate" , END)

# Compile
graph = workflow.compile()