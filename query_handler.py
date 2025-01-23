# from openai import OpenAI
# import faiss
# import numpy as np
# import json
# import os

# # Instantiate the OpenAI client
# client = OpenAI(api_key=os.getenv("API_KEY"))

# def query_faiss_index(user_query, index, documents, doc_map, top_k=3):
#     """
#     Query the FAISS index with the user query and retrieve the top-k relevant chunks.
#     """
#     # Generate embedding for the user query
#     response = client.embeddings.create(
#         model="text-embedding-ada-002",
#         input=user_query
#     )
#     query_embedding = np.array([response.data[0].embedding])

#     # Search the FAISS index
#     distances, indices = index.search(query_embedding, top_k)

#     # Retrieve matched chunks
#     results = []
#     for idx in indices[0]:
#         doc_id, chunk_id = doc_map[idx]
#         results.append({
#             "document": documents[doc_id]["name"],
#             "chunk": documents[doc_id]["chunks"][chunk_id]
#         })

#     return results

# def get_response_with_context(user_query, context):
#     """
#     Use OpenAI's ChatCompletion API to generate a response using the provided context.
#     """
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": "Use the following context to answer the question:"},
#             {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
#         ],
#         max_tokens=300
#     )
#     return response.choices[0].message.content

# # Example usage
# if __name__ == "__main__":
#     # Example setup and execution...
#     pass

from openai import OpenAI
import faiss
import numpy as np
import json
import os

# Instantiate the OpenAI client
client = OpenAI(api_key=os.getenv("API_KEY"))

def query_faiss_index(user_query, index, documents, doc_map, top_k=3):
    """
    Query the FAISS index with the user query and retrieve the top-k relevant chunks.
    """
    # Generate embedding for the user query
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=user_query
    )

    # Convert response to dictionary using model_dump()
    query_embedding = np.array([response.model_dump()["data"][0]["embedding"]])

    # Search the FAISS index
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve matched chunks
    results = []
    for idx in indices[0]:
        doc_id, chunk_id = doc_map[idx]
        results.append({
            "document": documents[doc_id]["name"],
            "chunk": documents[doc_id]["chunks"][chunk_id]
        })

    return results

def get_response_with_context(user_query, context):
    """
    Use OpenAI's ChatCompletion API to generate a response using the provided context.
    """
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Use the following context to answer the question:"},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
        ],
        max_tokens=300
    )

    # Convert response to dictionary using model_dump()
    return response.model_dump()["choices"][0]["message"]["content"]

# Example usage
if __name__ == "__main__":
    # Paths to required files
    index_path = "document_index.faiss"
    documents_path = "processed_documents.json"

    # Load documents
    with open(documents_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    # Load FAISS index
    index = faiss.read_index(index_path)

    # Create document map
    doc_map = []
    for doc_id, doc in enumerate(documents):
        for chunk_id in range(len(doc["chunks"])):
            doc_map.append((doc_id, chunk_id))

    # User query
    user_query = "What are the best practices for mental health counseling?"

    # Query the FAISS index
    results = query_faiss_index(user_query, index, documents, doc_map)

    # Combine relevant chunks into context
    context = "\n".join([res["chunk"] for res in results])

    # Generate response using context
    response = get_response_with_context(user_query, context)

    # Output the response
    print("Response from assistant:")
    print(response)



import os
import openai

# Retrieve the API key from the environment variable
api_key = os.getenv('API_KEY')

if api_key is None:
    raise ValueError("No API key found. Please set the API_KEY environment variable.")

# Set the API key for the OpenAI client
openai.api_key = api_key

# Initialize conversation history
conversation_history = [
    {"role": "system", "content": 
     "You are Wolfe, a compassionate and professional supportive AI therapist. Use the provided context to guide your response. "
                          "the user's feelings, be conversational, ask thoughtful questions that can let you understand them and provide them responses that"
                           "can help them feel comfortable with you, trust you, and open up more"
                            "this relationship should be quite established before you begin to provide thoughtful guidance based on the context. Keep your response warm,"
                             "empathetic, and non-judgmental."
                             "without making assumptions or adding unnecessary context"
                              "Keep your responses concise,"
                                "The context contains scripts and techniques used by professional therapists. Your role is to provide thoughtful guidance based on the context. "
                                "Keep your responses warm, empathetic, and concise."
                                "Aim to align responses more closely with the urgency and tone of the client’s statements, especially when frustration or a need for actionable help is expressed"
                                "key points to remember - Core tenets of therapy sessions generally include: building a trusting relationship with the therapist, active listening, empathy, confidentiality, client autonomy, goal setting, exploring emotions and thoughts, identifying patterns of behavior, and working collaboratively to develop coping mechanisms and strategies to address the client's concerns"
                                "dont ask for specifics repeatedly, especiallyw hen you are yet to unpack and dive into an initial specific detail client has shared with you" 
                                "when client shares challenges, you can delve deeper into their emotional or systemic roots" 
                                "dont overemphasize on external solutions especially when client say they have no trusted individual, counselors or communities, you are meant to be their professional therapist like sigmund freud"
                                  "be creative and mature with your solutions, they should be tailored to the context"
}
]


def chat_with_wolfe(user_input, conversation_history):
    """
    Handles back-and-forth conversation with Wolfe by appending user input and AI responses
    to the conversation history.
    """
    # Add user input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Generate Wolfe's response using OpenAI's Chat API
    response = openai.chat.completions.create(
        model="gpt-4o",
        max_tokens=150,
        temperature=0.3,
        messages=conversation_history
    )

    # Extract Wolfe's response
    ai_response = response.choices[0].message.content  # Corrected response handling

    # Add Wolfe's response to the conversation history
    conversation_history.append({"role": "assistant", "content": ai_response})

    # Return Wolfe's response
    return ai_response



if __name__ == "__main__":
    print("Wolfe: Hi there! How are you feeling today?")

    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            print("Wolfe: Take care of yourself. Goodbye!")
            break

        # Get Wolfe's response
        ai_response = chat_with_wolfe(user_input, conversation_history)

        # Print Wolfe's response
        print(f"Wolfe: {ai_response}")
