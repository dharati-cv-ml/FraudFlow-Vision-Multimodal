"""OPTIONAL direct-in-process VLM runner. No local server or localhost.

Requires separate installation of Torch + Transformers and a first-time model download.
Untested in the user's Python 3.14 Windows environment; run a small smoke sample.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

MODEL_DEFAULT='HuggingFaceTB/SmolVLM2-500M-Video-Instruct'
QUESTIONS="""Answer these five questions about the DOCUMENT IN THE IMAGE only.
Return exactly ONE JSON object with the keys:
"document_type" (receipt or invoice),
"date" (printed YYYY-MM-DD date),
"total" (printed TOTAL currency amount, e.g. $123.45),
"payment_reference" (printed ACCT-12345 identifier),
"arithmetic_consistency" (YES or NO: does the printed TOTAL equal subtotal plus tax within one cent?).
Never invent unreadable values; use an empty string for uncertainty. No prose or markdown."""

class LocalVLM:
    def __init__(self,model_name=MODEL_DEFAULT):
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError as e:
            raise RuntimeError('Install optional deps: python -m pip install -r requirements-vlm.txt; Torch version and platform support must be checked first') from e
        self.torch=torch
        self.model_name=model_name
        self.processor=AutoProcessor.from_pretrained(model_name)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.model=AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.float16 if self.device=='cuda' else torch.float32).to(self.device).eval()

    def answer(self, image_path):
        # Prefer the documented SmolVLM2 multimodal chat-template interface.
        from PIL import Image
        image=Image.open(image_path).convert('RGB')
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':QUESTIONS}]}]
        prompt=self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs=self.processor(text=prompt,images=[image],return_tensors='pt').to(self.device)
        with self.torch.inference_mode():
            result=self.model.generate(**inputs,max_new_tokens=190,do_sample=False)
        output_ids=result[0][inputs['input_ids'].shape[-1]:]
        raw=self.processor.decode(output_ids,skip_special_tokens=True)
        match=re.search(r'\{.*?\}',raw,re.S)
        if not match:
            return {},raw
        try:
            response=json.loads(match.group(0))
            return response if isinstance(response,dict) else {},raw
        except json.JSONDecodeError:
            return {},raw
