import json
from sentence_transformers import SentenceTransformer

# 1. Load the raw scraped data
raw_file = "raw_data.json"
print(f"Loading data from {raw_file}...")
with open(raw_file, "r", encoding="utf-8") as f:
    database = json.load(f)

# 2. Initialize the MiniLM model 
print("Loading paraphrase-multilingual-MiniLM-L12-v2 (this might take a moment)...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

total_chunks = 0

print("Grouping by 3, vectorizing, and rounding to 4 decimal places...")
for item in database:
    # Create a new empty list inside the existing item to hold our vectors
    item["chunks"] = []
    
    # Combine the positive matches into a single list
    lines_to_embed = item["covers"] + item["also_covers"] + item["examples"]
    
    # Group the lines into chunks of 3
    chunk_size = 3
    for i in range(0, len(lines_to_embed), chunk_size):
        # Grab up to 3 lines from the list
        current_group = lines_to_embed[i:i+chunk_size]
        
        # Clean up the lines (strip trailing periods so we don't get double punctuation)
        clean_group = [line.strip().rstrip('.') for line in current_group if line.strip()]
        
        # Skip if the group is somehow empty
        if not clean_group:
            continue
            
        # Join the 3 items together with a period and a space
        chunk_text = ". ".join(clean_group) + "."
        
        # Generate the 384-dimension vector from the grouped text
        vector = model.encode(chunk_text).tolist()
        
        # ROUNDING: Truncate every float in the vector to 4 decimal places
        rounded_vector = [round(num, 4) for num in vector]
        
        # Append the grouped text snippet and its rounded vector
        item["chunks"].append({
            "text": chunk_text,
            "vector": rounded_vector
        })
        total_chunks += 1
        
    print(f"Processed {item['sni']}: added {len(item['chunks'])} grouped vectors.")

# 4. Save the final payload
output_file = "embedded_database.json"
print(f"\nSaving database with {total_chunks} total vectors to {output_file}...")

# MINIFICATION: We use separators=(',', ':') to strip all unnecessary whitespace
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(database, f, separators=(',', ':'), ensure_ascii=False)

print("Done! Check your new, drastically smaller JSON file size.")