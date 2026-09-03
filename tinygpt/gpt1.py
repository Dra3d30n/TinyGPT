import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

MODEL = "openai-community/openai-gpt"

print("Loading GPT-1...")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

print(f"Loaded GPT-1 on {device}")
print("Type 'exit' to quit.\n")

while True:
    prompt = input("You: ")

    if prompt.lower() == "exit":
        break

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.00001,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    input_length = inputs["input_ids"].shape[1]

    generated = tokenizer.decode(
        output[0][input_length:],
        skip_special_tokens=True,
    )

    print(f"GPT-1: {generated.strip()}\n")


