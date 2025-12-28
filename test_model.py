# =====================================================
# CPU SAFETY
# =====================================================
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
import re

# =====================================================
# PATHS
# =====================================================
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = os.path.abspath("tinyllama_lora_adapter")

# =====================================================
# LOAD MODEL
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# =====================================================
# JSON EXTRACT
# =====================================================
def extract_json(text):
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except:
        return m.group()

# =====================================================
# FEW-SHOT INFERENCE (KEY FIX)
# =====================================================
def run_inference(review):
    prompt = f"""
Review: Battery is good but charging is slow.

### OUTPUT ###
{{"summary":["Good battery","Slow charging"],"pros":["good battery life"],"cons":["slow charging"],"sentiment":"mixed","verdict":"Good but has drawbacks."}}

Review: {review}

### OUTPUT ###
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            num_beams=1,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nRAW OUTPUT:\n", decoded)

    print("\nEXTRACTED JSON:\n", extract_json(decoded))

# =====================================================
# TEST
# =====================================================
run_inference("Battery is good but charging is slow.")
