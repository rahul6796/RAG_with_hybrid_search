import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dotenv import load_dotenv
import os
load_dotenv()

hug_api_key = os.getenv('HUGF_KEYS')
from huggingface_hub import login
login(token =hug_api_key)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "HuggingFaceH4/zephyr-7b-beta"

model = AutoModelForCausalLM.from_pretrained(
    model_id, load_in_8bit=False,trust_remote_code=True, device_map='mps'
)
