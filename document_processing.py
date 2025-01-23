from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import os

def extract_text_from_html(file_path):
    """
    Extract text from an HTML file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    return soup.get_text()

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF file.
    """
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_documents(base_directory):
    """
    Extract text from all HTML and PDF files in the specified base directory.
    Handles separate folders for HTML and PDF files.
    """
    documents = []

    # Define paths for HTML and PDF folders
    html_folder = os.path.join(base_directory, "HTML_Version")  # HTML files
    pdf_folder = os.path.join(base_directory, "PDF_Version")  # PDF files

    # Process HTML files
    if os.path.exists(html_folder):
        for file_name in os.listdir(html_folder):
            if file_name.endswith(".html"):
                file_path = os.path.join(html_folder, file_name)
                text = extract_text_from_html(file_path)
                documents.append({"name": file_name, "content": text})

    # Process PDF files
    if os.path.exists(pdf_folder):
        for file_name in os.listdir(pdf_folder):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(pdf_folder, file_name)
                text = extract_text_from_pdf(file_path)
                documents.append({"name": file_name, "content": text})

    return documents  # Move `return` outside the loop

def split_text_into_chunks(text, chunk_size=500):
    """
    Split the text into chunks of specified size (default: 500 tokens).
    """
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Example usage
if __name__ == "__main__":
    # Define the base directory where "HTML_Version" and "PDF_Version" are located
    base_directory = "Therapist_Scripts_documents"

    # Extract text from all documents
    documents = extract_text_from_documents(base_directory)

    # Split the content of each document into chunks
    for doc in documents:
        doc["chunks"] = split_text_into_chunks(doc["content"])

    # Print summary
    print(f"Processed {len(documents)} documents.")
    for doc in documents:
        print(f"Document: {doc['name']}, Number of Chunks: {len(doc['chunks'])}")

        import json

# Save processed data to a JSON file
with open("processed_documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False, indent=4)

print("Processed data saved to 'processed_documents.json'")

