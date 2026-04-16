import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCustomTasks
from huggingface_hub import login

# 1. Authenticate with Hugging Face using your read-only token
HF_TOKEN = "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"
login(token=HF_TOKEN)

# 2. Initialize the ONNX model exactly as we did in the test script
repo_id = "Kristian-E/sentence-bert-swedish-onnx"
print(f"Downloading/Loading model from {repo_id}...")
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = ORTModelForCustomTasks.from_pretrained(repo_id, file_name="model_quantized.onnx")

def get_embedding(text):
    """Helper function to calculate the exact mean-pooled vector."""
    encoded_input = tokenizer([text], padding=True, truncation=True, max_length=384, return_tensors='pt')
    
    with torch.no_grad():
        model_output = model(**encoded_input)
        
    # Extract tensor dynamically
    token_embeddings = model_output.get("token_embeddings", model_output.get("last_hidden_state"))
    if token_embeddings is None:
        token_embeddings = list(model_output.values())[0]
        
    # Apply mean pooling
    attention_mask = encoded_input['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    
    # L2 Normalization
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # Return as a simple python list and round to 4 decimal places
    vector = embeddings[0].tolist()
    return [round(num, 4) for num in vector]

# 3. Load the raw scraped data
raw_file = "raw_data.json" # Assuming testing raw data based on your file structure
print(f"\nLoading data from {raw_file}...")
with open(raw_file, "r", encoding="utf-8") as f:
    database = json.load(f)

total_chunks = 0
print("Extracting title + first 3 covers, vectorizing, and rounding...")

for item in database:
    item["chunks"] = []
    
    # Extract Title (safely)
    title = item.get("title", "").strip().rstrip('.')
    
    # Extract first 3 lines of 'covers'
    covers = item.get("covers", [])
    first_three_covers = covers[:3]
    clean_covers = [line.strip().rstrip('.') for line in first_three_covers if line.strip()]
    
    # Combine them
    parts = []
    if title:
        parts.append(title)
    parts.extend(clean_covers)
    
    if not parts:
        continue # Skip if completely empty
        
    chunk_text = ". ".join(parts) + "."
    
    # Generate the vector
    rounded_vector = get_embedding(chunk_text)
    
    item["chunks"].append({
        "text": chunk_text,
        "vector": rounded_vector
    })
    total_chunks += 1
        
    print(f"Processed {item.get('sni', 'Unknown')}: {chunk_text[:50]}...")

# 4. Save the final payload
output_file = "embedded_database.json"
print(f"\nSaving database with {total_chunks} total vectors to {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(database, f, separators=(',', ':'), ensure_ascii=False)

print("Done! Check your new, drastically smaller JSON file size.")