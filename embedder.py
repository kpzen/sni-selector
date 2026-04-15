import json
from sentence_transformers import SentenceTransformer

# 1. Load the raw scraped data
raw_file = "raw_data_test.json"
print(f"Loading data from {raw_file}...")
with open(raw_file, "r", encoding="utf-8") as f:
    database = json.load(f)

# 2. Initialize the model 
print("Loading E5 model (this might take a moment)...")
model = SentenceTransformer('intfloat/multilingual-e5-small')

total_chunks = 0

print("Chunking and vectorizing line-by-line...")
for item in database:
    # Create a new empty list inside the existing item to hold our vectors
    item["chunks"] = []
    
    # Combine the positive matches into a single list to iterate through
    # We strictly EXCLUDE 'does_not_cover' so the AI doesn't learn negated concepts
    lines_to_embed = item["covers"] + item["also_covers"] + item["examples"]
    
    for line in lines_to_embed:
        # The E5 model requires the 'passage: ' prefix for the database side
        text_to_embed = f"passage: {line}"
        
        # Generate the 384-dimension vector
        vector = model.encode(text_to_embed).tolist()
        
        # Append the text snippet and its vector directly into the parent item
        item["chunks"].append({
            "text": line,
            "vector": vector
        })
        total_chunks += 1
        
    print(f"Processed {item['sni']}: added {len(item['chunks'])} vectors.")

# 4. Save the final payload
output_file = "embedded_database.json"
print(f"\nSaving database with {total_chunks} total vectors to {output_file}...")

# We use indent=2 to keep the JSON highly readable (pretty-printed)
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(database, f, indent=2, ensure_ascii=False)

print("Done! Check your beautifully formatted JSON file.")