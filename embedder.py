import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCustomTasks

# 1. Initialize the ONNX model and tokenizer
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

# 2. Load the raw scraped data
raw_file = "raw_data.json" # Change back to raw_data.json if needed, using your standard output
if not os.path.exists(raw_file):
    # Fallback to test data if raw_data.json isn't found
    raw_file = "raw_data_test.json" 

print(f"\nLoading data from {raw_file}...")
with open(raw_file, "r", encoding="utf-8") as f:
    database = json.load(f)

total_chunks = 0
MAX_TOKENS = 380

print("Extracting covers, applying smart punctuation, and chunking by token limit...")

for item in database:
    item["chunks"] = []
    
    # Extract and clean the title
    title = item.get("title", "").strip().rstrip('.:,;')
    if not title:
        continue # Skip if there's no title
        
    title_str = f"{title}."
    title_tokens = len(tokenizer.encode(title_str, add_special_tokens=False))
    
    # Combine covers and also_covers (ignoring examples)
    lines_to_embed = item.get("covers", []) + item.get("also_covers", [])
    
    current_chunk_lines = [title_str]
    current_tokens = title_tokens
    
    for raw_line in lines_to_embed:
        # Clean up existing punctuation to prevent double-punctuation
        line = raw_line.strip().rstrip('.:,;')
        if not line:
            continue
            
        # Smart Punctuation: Single words get a comma, phrases get a period
        word_count = len(line.split())
        if word_count == 1:
            formatted_line = f"{line},"
        else:
            formatted_line = f"{line}."
            
        line_tokens = len(tokenizer.encode(formatted_line, add_special_tokens=False))
        
        # Check if adding this line exceeds our safe token limit
        # Ensure we don't accidentally strand a single line that is longer than MAX_TOKENS by itself
        if current_tokens + line_tokens > MAX_TOKENS and len(current_chunk_lines) > 1:
            # 1. Finalize the current chunk
            chunk_text = " ".join(current_chunk_lines)
            
            # If the chunk ends with a comma, replace it with a period
            if chunk_text.endswith(','):
                chunk_text = chunk_text[:-1] + '.'
                
            # 2. Vectorize and save
            rounded_vector = get_embedding(chunk_text)
            item["chunks"].append({
                "text": chunk_text,
                "vector": rounded_vector
            })
            total_chunks += 1
            
            # 3. Start a new chunk, resetting with the title
            current_chunk_lines = [title_str, formatted_line]
            current_tokens = title_tokens + line_tokens
        else:
            # Fits in the current chunk, so just add it
            current_chunk_lines.append(formatted_line)
            current_tokens += line_tokens
            
    # Process the final leftover chunk if it has data
    if current_chunk_lines:
        chunk_text = " ".join(current_chunk_lines)
        
        # Clean trailing comma again just in case the very last line was a single word
        if chunk_text.endswith(','):
            chunk_text = chunk_text[:-1] + '.'
            
        rounded_vector = get_embedding(chunk_text)
        item["chunks"].append({
            "text": chunk_text,
            "vector": rounded_vector
        })
        total_chunks += 1
        
    print(f"Processed {item.get('sni', 'Unknown')}: added {len(item['chunks'])} chunk(s).")

# 3. Save the final payload
output_file = "embedded_database.json"
print(f"\nSaving database with {total_chunks} total vectors to {output_file}...")

with open(output_file, "w", encoding="utf-8") as f:
    # Minifying the JSON heavily to save file size
    json.dump(database, f, separators=(',', ':'), ensure_ascii=False)

print("Done! Ready to bundle.")