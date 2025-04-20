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
            work_process = st.text_input("**✅ 작업 프로세스**", value = "ex) 절단작업, 타설작업")
            accident_object = st.text_input("**✅ 사고객체**", value = "ex) 공구류")
            human_accident = st.text_input("**✅ 인적사고**", value = "ex) 끼임")
            property_accident = st.text_input("**✅ 물적사고**", value = "ex) 없음")
            
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
        combined_prompt = f"""
        {prompt} 로 인해 사고가 발생했습니다.
        해당 사고는 {work_process} 중 발생했으며, 관련 사고 객체는 {accident_object} 입니다.
        이로 인한 인적피해는 {human_accident} 이고, 물적피해는 {property_accident} 로 확인됩니다.
        재발 방지 대책 및 향후 조치 계획은 무엇인가요?
        """
        
        st.chat_message("user").write(combined_prompt)

        # AI 응답처리 
        with st.chat_message("assistant") : 
            
            # 초기 상태 설정
            initial_state = {
                "question" : combined_prompt , 
                "pdf_docs" : [],
                "csv_docs" : [] , 
                "generation" : ""
            }
            
            try : 
                # 그래프 실행 및 상태 업데이트 (노드 실행결과를 step 으로 받아 줌)
                for step in graph.stream(initial_state) :
                    
                    # 현재 단계 표시(node_name : 노드 이름 , state : 노드 결과값 )
                    for node_name , state in step.items() : 
                        
                        if node_name == "retrieve": 
                            with st.expander("👷🏼 예시 검색 결과") : 
                                for i , result in enumerate(state["csv_docs"]) :
                                    st.write(f"Source {i} : {result}")
                        
                        if node_name == "grade_documents":  # 필터링 된 pdf 문서 
                            with st.expander("🔍 PDF 검색 결과") :
                                for i , result in enumerate(state["pdf_docs"]) :
                                    st.write(f"Source {i} : {result}") 
                        
                        if "generation" in state :
                            last_msg = state["generation"]
                            st.session_state.messages.append(AIMessage(content = last_msg))
                            st.markdown(last_msg)
                            
            except Exception as e :
                st.error(f"Error 발생 : {str(e)}")

    # 채팅 기록 초기화 버튼 
    if st.button("대화 기록 지우기") : 
        
        st.session_state.messages = [
            AIMessage(content = "안녕하세요! 건설사고 대응책 AI 어시스턴트입니다. 사고 상황에 대해 설명해주세요! ")
        ]
        
        # 사이드바 입력값 초기화
        keys_to_clear = ["work_process", "accident_object", "human_accident", "property_accident"]
        
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]  
                
        st.rerun() # 페이지 새로고침 
    
    
if __name__ == "__main__" :
    main()