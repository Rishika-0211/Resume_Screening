import fitz  # pymupdf
import os
import json
import csv
from transformers import pipeline

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Step 2: Load categories from categories.json
def load_categories():
    with open("categories.json", "r") as f:
        data = json.load(f)
    return data["categories"]

# Step 3: Load LLM classifier
def load_model():
    print("🔄 Loading zero-shot classification model...")
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Step 4: Ask the model to classify
def classify_with_model(classifier, text, categories):
    prediction = classifier(text, candidate_labels=categories)
    top_predictions = list(zip(prediction["labels"][:3], prediction["scores"][:3]))
    return top_predictions

# Step 5: Save to CSV
def save_to_csv(results, filename="results.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Resume", "Top 3 Predicted Roles"])
        
        for resume_file, predictions in results:
            formatted = "; ".join([f"{label} ({score:.2f})" for label, score in predictions])
            writer.writerow([resume_file, formatted])

# Step 6: Main process
def main():
    print("🚀 Starting model-based Resume Screening Assistant...")

    classifier = load_model()
    print("✅ Device set to use CPU")

    resume_folder = "resumes"  # Folder containing your PDF resumes
    categories = load_categories()  # Load categories from JSON
    results = []

    for file_name in os.listdir(resume_folder):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(resume_folder, file_name)
            print(f"📄 Processing {file_name}...")

            text = extract_text_from_pdf(file_path)
            prompt = f"Classify this resume content:\n{text[:1000]}"

            top_predictions = classify_with_model(classifier, prompt, categories)
            results.append((file_name, top_predictions))

            print(f"✅ {file_name} → {top_predictions[0][0]}")

    save_to_csv(results)
    print("📄 Done! Results saved to results.csv")

# Run the script
if __name__ == "__main__":
    main()
