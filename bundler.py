import json
import os

print("Starting bundler...")

# 1. Read the HTML template
print("Reading template.html...")
with open("template.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# 2. Read the embedded JSON
print("Reading embedded_database.json...")
with open("embedded_database.json", "r", encoding="utf-8") as f:
    database_content = f.read()

# 3. Inject the data into the HTML
print("Injecting database into HTML...")
final_html = html_content.replace("/*INJECT_DATABASE_HERE*/", database_content)

# 4. Save the final deliverable file as index.html for GitHub Pages
output_file = "index.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(final_html)

print(f"Success! Your portable demo is ready to be hosted: {output_file}")