import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_llm(model_id = "Qwen/QwQ-32B") : 

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",  
            "bnb_4bit_compute_dtype": torch.bfloat16  
        }
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def generate_response(llm_model, tokenizer, pdf_docs, csv_docs, question):

    llm = llm_model

    system_message = (
     "<|im_start|>system\n"
      "당신은 건설 안전 사고 분석 및 예방 대책 수립에 전문성을 갖춘 안전관리 전문가입니다.\n"
      "당신의 임무는 발생한 사고에 대해 단계적으로 분석하고 사고의 주요 위험 요소와 원인을 파악하여, "
      "재발 방지를 위한 대책 및 향후 조치 계획을 수립하는 것입니다.\n"
      "가장 적합한 대응책을 생성하기 위해서 유사 사고 사례와 관련 안전 지침서를 활용하여 체계적인 추론을 통해서 종합적으로 판단해야합니다.\n"
      "사고의 구조와 맥락을 단계적으로 사고한 후 결론을 도출하세요.\n"
      "답변은 반드시 한국어로 작성해야 합니다.\n"
      "<|im_end|>"
    )
    
    user_message = (
        "<|im_start|>user\n"
        f"[발생한 사고] : {question}\n"
        "[유사 사고 사례 및 대응책]\n"
        f"{csv_docs}\n"
        "[사고 관련 안전 지침서]\n"
        f"{pdf_docs}\n"
        "위의 유사사고사례와 관련 안전 지침서를 참고하여, 발생한 사고에 대한 최적의 재발방지 대책 및 향후 조치 계획을 간결하게 작성해주세요.\n"
        "반드시 '재발 방지 대응책:' 으로 시작하는 2줄 이내의 간결한 문장으로 답변하세요.\n"
        "하나씩 차근차근 생각한 뒤, 최종 대응 대책을 도출하세요.\n"
        "<|im_end|>"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response 