from transformers import AutoTokenizer
from vllm import LLM , SamplingParams 
import torch 

def vllm_model(model_name = "LGAI-EXAONE/EXAONE-Deep-7.8B" , gpu_memory = 0.7 , max_model_len = 4096)
    
    model_id = model_name 

    llm = LLM(
    model_id = model_id , 
    trust_remote_code = True ,
    quantization = "bitsandbytes" ,
    dtype = torch.bfloat16,
    gpu_memory_utilization = gpu_memory,
    max_model_len = max_model_len
    ) 
    
    return llm 


def generate_prompt(model_name) :
    return 

def main() : 
    
    data = pd.read_csv('../test_data.csv')

    model_id = "LGAI-EXAONE/EXAONE-Deep-7.8B"
    col_name = model_id.split('/')[-1]

    llm = vllm_model(model_name = model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_id).eos_token_id
    eos_token_id = tokenizer.eos_token_id

    result_text = []

    llm.llm_engine.scheduler_config.max_num_seqs = 8 

    sampling_params = SamplingParams(temperature = 0.2 , 
                                    top_p = 0.8 , 
                                    seed = 42 , 
                                    max_tokens = 512 , 
                                    stop_token_ids = [eos_token_id])

    outputs = llm.chat(messages, sampling_params)

    for output in outputs :
        generated_text = output.outputs[0].text
        generated_text = generated_text.strip()
        result_text.append(generated_text)
    
    test_data[col_name] = result_text
    
    print(f'{col_name} 모델 응답 추론 완료')








