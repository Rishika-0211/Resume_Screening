import fitz  # PyMuPDF
import os
import json
import csv
from collections import defaultdict

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Save extracted text
def save_extracted_text(filename, text):
    output_path = os.path.join("extracted_texts", filename.replace(".pdf", ".txt"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

# Step 3: Classify resume
def classify_resume(text, categories):
    scores = defaultdict(int)
    lower_text = text.lower()
    for category, keywords in categories.items():
        for kw in keywords:
            if kw.lower() in lower_text:
                scores[category] += 1
    if scores:
        return max(scores, key=scores.get)
    return "Other"

# Step 4: Save results to CSV
def save_to_csv(results, filename="results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Resume", "Predicted Category"])
        for res in results:
            writer.writerow(res)

# Step 5: Main execution
def main():
    print("✅ Script started")

    with open("categories.json") as f:
        categories = json.load(f)
        print("📁 Categories loaded:", categories)

    results = []
    for file in os.listdir("resumes"):
        print(f"📄 Found file: {file}")
        if file.endswith(".pdf"):
            path = os.path.join("resumes", file)
            text = extract_text_from_pdf(path)

            save_extracted_text(file, text)

            predicted = classify_resume(text, categories)
            print(f"✅ {file} → {predicted}")
            results.append((file, predicted))
    
    save_to_csv(results)
    print("📄 results.csv generated.")

main()
