from transformers import AutoTokenizer, AutoModelForCausalLM

print("Downloading Mistral-7B-Instruct-v0.2 model...")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
print("Download complete ✅")
