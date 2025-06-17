import fitz  # PyMuPDF
import os

pdf_folder = "data/pdfs"  # where your PDFs are
output_file = "data/combined_text.txt"

def extract_text_from_pdfs(folder_path):
    combined_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            doc = fitz.open(path)
            for page in doc:
                combined_text += page.get_text()
            print(f"✅ Extracted: {filename}")
    return combined_text

if __name__ == "__main__":
    text = extract_text_from_pdfs(pdf_folder)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n📄 Done! Text saved to {output_file}")
