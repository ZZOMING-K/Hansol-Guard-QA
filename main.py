import os 
import streamlit as st
from langchain_core.messages import AIMessage , HumanMessage
from agent_graph import graph 

def main() : 
    
    st.set_page_config(page_title="Hansol Guard", 
                        page_icon=":construction:", 
                        layout="centered",)
        
    st.title("Hansol Guard QA ChatBot :construction:")

    with st.sidebar : 
        st.subheader("사고 정보 입력란")
        st.markdown(" ")
        work_clf = st.text_input("**✅ 공종**", placeholder = "ex) 철큰콘크리트공사")
        work_process = st.text_input("**✅ 작업 프로세스**", placeholder = "ex) 절단작업, 타설작업")
        accident_object = st.text_input("**✅ 사고객체**", placeholder = "ex) 공구류")
        human_accident = st.text_input("**✅ 인적사고**", placeholder = "ex) 끼임")
        property_accident = st.text_input("**✅ 물적사고**", placeholder = "ex) 없음")
            
    # 세션상태 초기화 
    if 'message' not in st.session_state : 
        st.session_state["messages"] = [
            AIMessage(content = "안녕하세요! 건설사고 대응책 AI 어시스턴트입니다. 사고 상황에 대해 설명해주세요! ")
        ]
        
    # 메세지 히스토리 표시 
    for msg in st.session_state.messages :
        if isinstance(msg , AIMessage) : 
            st.chat_message("assistant").write(msg.content)
        if isinstance(msg , HumanMessage) :
            st.chat_message("user").write(msg.content)
            
    # 사용자 입력 처리 
    if prompt := st.chat_input("사고 원인을 설명해주세요.") : 
        st.session_state.messages.append(HumanMessage(content = prompt)) 
        
        # 사고 정보와 결합 
        accicdent_prompt = f"""
        공종 : {work_clf}
        작업프로세스 : {work_process}
        사고객체 : {accident_object}
        인적사고 : {human_accident}
        물적사고 : {property_accident}
        사고원인 : {prompt}\n
        위 사고 상황에 대한 재발방지 대책 및 향후조치계획은 무엇인가요? 
        """
        
        st.chat_message("user").write(accicdent_prompt)
        
        question_list = [ work_clf, work_process, accident_object, human_accident, property_accident, prompt ]

        # AI 응답처리 
        with st.chat_message("assistant") : 
            
            # 초기 상태 설정
            initial_state = {
                # 공종, 작업프로세스, 사고객체, 인적사고, 물적사고, 사고원인
                "question" : question_list , 
                "pdf_prompt" : "" ,  
                "csv_prompt" : "" ,
                "pdf_docs" : [] ,
                "csv_docs" : [] , 
                "think_response" : "",
                "final_response" : "",
                "generator" : None
            }
            
            thinking_placeholder = st.empty()
            response_placeholder = st.empty()
            
            try : 
                # 그래프 실행 및 상태 업데이트 (노드 실행결과를 step 으로 받아 줌)
                for step in graph.stream(initial_state) :
                    
                    # 현재 단계 표시(node_name : 노드 이름 , state : 노드 결과값 )
                    for node_name , state in step.items() : 
                        
                        if node_name == "retrieve": 
                            with st.expander("👷🏼 예시 검색 결과") : 
                                for i , result in enumerate(state["csv_docs"]) :
                                    st.write(f"{result}")
                                    
                            with st.expander("🔍 PDF 검색 결과") :
                                for i , result in enumerate(state["pdf_docs"]) :
                                    st.write(f"{result}") 
                        
                        if node_name == "generate":
                            if "generator" in state:
                                
                                generator = state["generator"]
                                
                                thinking_content = ""
                                response_content = ""
                                
                                for chunk in generator(): #streaming
                                    thinking_content = chunk["think_response"]
                                    response_content = chunk["final_response"]
                                    
                                    with st.expander("💭 사고 과정", expanded=False):
                                        thinking_placeholder.markdown(thinking_content)
                                    
                                    # 최종 응답 업데이트 (mode_change가 True일 때 표시 시작)
                                    if chunk["mode_change"] or response_content:
                                        response_placeholder.markdown(response_content)
                                
                                # 최종 응답 저장
                                st.session_state.messages.append(AIMessage(content=response_content))
                                
            except Exception as e :
                st.error(f"Error 발생 : {str(e)}")

    # 채팅 기록 초기화 버튼 
    if st.button("대화 기록 지우기") : 
        
        st.session_state.messages = [
            AIMessage(content = "안녕하세요! 건설사고 대응책 AI 어시스턴트입니다. 사고 상황에 대해 설명해주세요! ")
        ]
        
        # 사이드바 입력값 초기화
        keys_to_clear = ["work_clf" , "work_process", "accident_object", "human_accident", "property_accident"]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]  
                
        st.rerun() # 페이지 새로고침 
    
    
if __name__ == "__main__" :
    main()