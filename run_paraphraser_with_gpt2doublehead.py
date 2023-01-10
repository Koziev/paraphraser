"""
Экспериментальный код генерации перефразировок с моделью, обученной в train_paraphraser_with_gpt2doublehead.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "inkoziev/paraphraser"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)
model.eval()

while True:
    seed = input(':> ').strip()
    encoded_prompt = tokenizer.encode("<s>" + seed + "<sep>", add_special_tokens=False, return_tensors="pt").to(device)
    output_sequences = model.generate(input_ids=encoded_prompt,
                                      max_length=100,
                                      typical_p=0.85,
                                      top_k=0,
                                      top_p=1.0,
                                      do_sample=True,
                                      num_return_sequences=10,
                                      pad_token_id=tokenizer.pad_token_id)

    for o in output_sequences:
        text = tokenizer.decode(o.tolist(), clean_up_tokenization_spaces=True)
        text = text[text.index('<sep>') + 5:]
        text = text[: text.find('</s>')]
        print(text)
