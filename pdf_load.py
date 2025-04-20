# 필요한 라이브러리 import
import torch
import base64
import urllib.request
from io import BytesIO
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text
from PyPDF2 import PdfReader
import json
import os 
import glob

# model load 
model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def ocr_pdf_page(pdf_path, page_number):

    # Render page 1 to an image
    image_base64 = render_pdf_to_base64png(pdf_path, page_number, target_longest_image_dim=1024)

    # Build the prompt, using document metadata
    anchor_text = get_anchor_text(pdf_path, 1, pdf_engine="pdfreport", target_length=4000)
    prompt = build_finetuning_prompt(anchor_text)

    # Build the full prompt
    messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]

  
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for (key, value) in inputs.items()}


    # Generate the output
    output = model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=2048,
                num_return_sequences=1,
                do_sample=True,
            )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )

    return text_output[0]


def get_pdf_page_count(pdf_path):
    
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    
    except Exception as e:
        print(f"Error using PyPDF2: {e}")
        return None


pdf_list = glob.glob('./data/pdf/*.pdf')


for idx , pdf_file in enumerate(pdf_list) :

    FILE_PATH = pdf_file
    NUM_PAGES = get_pdf_page_count(FILE_PATH)

    pages = []

    for page_idx in range(NUM_PAGES):
        print(f"Processing page {page_idx+1} of {NUM_PAGES}")
        page_json = ocr_pdf_page(FILE_PATH, page_idx+1)
        pages.append(page_json)

    file_name = os.path.splitext(os.path.basename(FILE_PATH))[0] # 파일 이름
    save_path = './pdf2txt/' # 파일 저장할 경로 설정
    output_file = os.path.join(save_path, file_name + ".md") #전체 경로

    # 파일 열기 (쓰기 모드)
    with open(output_file, "w", encoding="utf-8") as f:

        for page_json in pages:
            try:
                data = json.loads(page_json)
                f.write(data['natural_text'] + "\n\n")  # 원본 그대로 저장
            except Exception as e:
                f.write(f"Error processing page_json: {str(e)}\n\n")

    print(f"{idx+1}/{len(pdf_list)} 문서 변환 완료 : Data saved to {output_file}")