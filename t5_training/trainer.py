from transformers import T5Tokenizer
from transformers import T5EncoderModel, T5Tokenizer
from dataset_utils import T5Dataset, collator
from torch.utils.data import DataLoader
import torch
import os

model_variants = ["google-t5/t5-base", "google-t5/t5-3b", "google/flan-t5-large", "google-t5/t5-11b"]
device = 'cuda:0'

model = T5EncoderModel.from_pretrained(
    model_variants[0],
    torch_dtype=torch.float16,
    token = os.getenv('HF_ACCESS_TOKEN')
).to(device)

tokenizer = T5Tokenizer.from_pretrained(
    model_variants[0]
)

raw_input = '/home/tadesa1/ADBMO-UNLV/data/processed_output_raw.txt'
with open(raw_input, 'r') as f:
    text = f.readlines()

dataset = T5Dataset(text, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda b: collator(b, tokenizer))