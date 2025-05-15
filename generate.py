import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Tread

def load_llm(model_id = "Qwen/QwQ-32B") : 

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",  
            "bnb_4bit_compute_dtype": torch.bfloat16  
        }
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    return model, tokenizer

def generate_response(llm_model, tokenizer, pdf_docs, csv_docs, question):

    llm = llm_model

    system_message = (
      "당신은 건설 안전 사고 분석 및 예방 대책 수립에 전문성을 갖춘 안전관리 전문가입니다.\n"
      "당신의 임무는 발생한 사고에 대해 단계적으로 분석하고 사고의 주요 위험 요소와 원인을 파악하여, "
      "재발 방지를 위한 대책 및 향후 조치 계획을 수립하는 것입니다.\n"
      "가장 적합한 대응책을 생성하기 위해서 유사 사고 사례와 관련 안전 지침서를 활용하여 체계적인 추론을 통해서 종합적으로 판단해야합니다.\n"
      "사고의 구조와 맥락을 단계적으로 사고한 후 결론을 도출하세요.\n"
      "유사 사고 사례 및 관련 안전 지침서에 없는 표기를 함부로 사용하지 마세요.\n"
      "**또한, 사고과정과 답변은 반드시 한국어로만 작성해야 합니다.**"
    )
    
    user_message = (
        f"[발생한 사고]\n"
        f"{question}\n"
        "[유사 사고 사례 및 대응책]\n"
        f"{csv_docs}\n"
        "[사고 관련 안전 지침서]\n"
        f"{pdf_docs}\n"
        "위의 유사사고사례와 관련 안전 지침서를 참고하여, 발생한 사고에 대한 최적의 재발방지 대책 및 향후 조치 계획을 간결하게 작성해주세요.\n"
        "반드시 '재발 방지 대응책:' 으로 시작하는 2줄 이내의 간결한 문장으로 답변하세요.\n"
        "하나씩 차근차근 생각한 뒤, 최종 대응 대책을 도출하세요."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
    )
    
    streamer = TextIteratorStreamer(tokenizer , skip_prompt = True , skip_special_tokens = True)
    
    
    def stream_generator() : 
    
        thread = Thread(target=model.generate, kwargs=dict(
            input_ids=input_ids.to("cuda"),
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=32768,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            streamer=streamer
        ))
    
        thread.start()
        
        think_model = True 
        think_response , final_response = "" , ""

        for text in streamer:
            if "</think>" in text:
                parts = text.split("</think>")
                if think_mode:
                    think_response += parts[0]
                    final_response += parts[1] if len(parts) > 1 else ""
                    think_mode = False
                else:
                    final_response += text
                
                yield {
                    "think_response": think_response, 
                    "final_response": final_response, 
                    "mode_change": True
                }
            else:
                if think_mode:
                    think_response += text
                else:
                    final_response += text
                
                yield {
                    "think_response": think_response, 
                    "final_response": final_response,
                    "mode_change": False
                }

    return stream_generator