from openai import OpenAI
import faiss
import numpy as np
import os

# Retrieve the API key from the environment variable
api_key = os.getenv('API_KEY')
if api_key is None:
    raise ValueError("No API key found. Please set the API_KEY environment variable.")

# Instantiate the OpenAI client
client = OpenAI(api_key=api_key)

def generate_embeddings(chunks):
    """
    Generate embeddings for a list of text chunks using OpenAI's embedding endpoint.
    """
    embeddings = []
    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)  # Updated access syntax
    return embeddings

# def create_faiss_index(documents):
#     """
#     Create a FAISS index from the document chunks and map document and chunk IDs.

#     Args:
#         documents (list): List of dictionaries, each containing "name" and "chunks".

#     Returns:
#         index (faiss.Index): The FAISS index for the embeddings.
#         doc_map (list): A mapping of document and chunk IDs.
#     """
#     index = None
#     doc_map = []

#     for doc_id, doc in enumerate(documents):
#         print(f"Generating embeddings for document: {doc['name']}")
#         embeddings = generate_embeddings(doc["chunks"])

#         if index is None:
#             # Initialize FAISS index with the embedding dimension
#             dimension = len(embeddings[0])
#             index = faiss.IndexFlatL2(dimension)

#         # Add embeddings to the FAISS index
#         index.add(np.array(embeddings))

#         # Map document ID and chunk ID
#         doc_map.extend([(doc_id, chunk_id) for chunk_id in range(len(doc["chunks"]))])


def create_faiss_index(documents):
    index = None
    doc_map = []
    for doc_id, doc in enumerate(documents):
        print(f"Generating embeddings for document: {doc['name']}")
        embeddings = generate_embeddings(doc["chunks"])

        if index is None:
            # Initialize FAISS index with the embedding dimension
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)

        # Add embeddings to the FAISS index
        index.add(np.array(embeddings))

        # Map document ID and chunk ID
        doc_map.extend([(doc_id, chunk_id) for chunk_id in range(len(doc["chunks"]))])

    return index, doc_map  # Make sure this is not indented inside the loop


 
if __name__ == "__main__":
    import json

    print("Loading processed documents...")
    # Load processed documents (replace with your actual file path)
    with open("processed_documents.json", "r", encoding="utf-8") as f:
        documents = json.load(f)

    print("Creating the FAISS index...")
    # Create the FAISS index
    index, doc_map = create_faiss_index(documents)

    print("Saving the FAISS index to a file...")
    # Save the FAISS index to a file
    faiss.write_index(index, "document_index.faiss")
    print("FAISS index saved to 'document_index.faiss'")
