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
                        # "The context contains scripts and techniques used by professional therapists. Your role is to validate"
                          "the user's feelings, be conversational, ask thoughtful questions that can let you understand them and provide them responses that"
                           "can help them feel comfortable with you, trust you, and open up more"
                            "this relationship should be quite established before you begin to provide thoughtful guidance based on the context. Keep your response warm,"
                             "empathetic, and non-judgmental."
                             "without making assumptions or adding unnecessary context"
                              "Keep your responses concise,"
                                "The context contains scripts and techniques used by professional therapists. Your role is to provide thoughtful guidance based on the context. "
                                "Keep your responses warm, empathetic, and concise."

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
        max_tokens=250,
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




















# import os
# import openai

# # Retrieve the API key from the environment variable
# api_key = os.getenv('API_KEY')

# if api_key is None:
#     raise ValueError("No API key found. Please set the API_KEY environment variable.")

# # Set the API key for the OpenAI client
# openai.api_key = api_key

# # Initialize conversation history
# conversation_history = [
#     {"role": "system", "content": 
#      "You are Wolfe, a compassionate and professional supportive AI therapist. Use the provided context to guide your response. "
#                           "the user's feelings, be conversational, ask thoughtful questions that can let you understand them and provide them responses that"
#                            "can help them feel comfortable with you, trust you, and open up more"
#                             "this relationship should be quite established before you begin to provide thoughtful guidance based on the context. Keep your response warm,"
#                              "empathetic, and non-judgmental."
#                              "without making assumptions or adding unnecessary context"
#                               "Keep your responses concise,"
#                                 "The context contains scripts and techniques used by professional therapists. Your role is to provide thoughtful guidance based on the context. "
#                                 "Keep your responses warm, empathetic, and concise."
#                                 "Aim to align responses more closely with the urgency and tone of the client’s statements, especially when frustration or a need for actionable help is expressed"
#                                 "key points to remember - Core tenets of therapy sessions generally include: building a trusting relationship with the therapist, active listening, empathy, confidentiality, client autonomy, goal setting, exploring emotions and thoughts, identifying patterns of behavior, and working collaboratively to develop coping mechanisms and strategies to address the client's concerns"
#                                 "dont ask for specifics repeatedly, especiallyw hen you are yet to unpack and dive into an initial specific detail client has shared with you" 
#                                 "when client shares challenges, you can delve deeper into their emotional or systemic roots" 
#                                 "dont overemphasize on external solutions especially when client say they have no trusted individual, counselors or communities, you are meant to be their professional therapist like sigmund freud"
#                                   "be creative and mature with your solutions, they should be tailored to the context"
# }
# ]


# def chat_with_wolfe(user_input, conversation_history):
#     """
#     Handles back-and-forth conversation with Wolfe by appending user input and AI responses
#     to the conversation history.
#     """
#     # Add user input to the conversation history
#     conversation_history.append({"role": "user", "content": user_input})

#     # Generate Wolfe's response using OpenAI's Chat API
#     response = openai.chat.completions.create(
#         model="gpt-4o",
#         max_tokens=150,
#         temperature=0.3,
#         messages=conversation_history
#     )

#     # Extract Wolfe's response
#     ai_response = response.choices[0].message.content  # Corrected response handling

#     # Add Wolfe's response to the conversation history
#     conversation_history.append({"role": "assistant", "content": ai_response})

#     # Return Wolfe's response
#     return ai_response



# if __name__ == "__main__":
#     print("Wolfe: Hi there! How are you feeling today?")

#     while True:
#         # Get user input
#         user_input = input("You: ")
#         if user_input.lower() in {"exit", "quit"}:
#             print("Wolfe: Take care of yourself. Goodbye!")
#             break

#         # Get Wolfe's response
#         ai_response = chat_with_wolfe(user_input, conversation_history)

#         # Print Wolfe's response
#         print(f"Wolfe: {ai_response}")


            

        




# def chat_with_wolfe(user_input, conversation_history):
#     """
#     Handles back-and-forth conversation with Wolfe by appending user input and AI responses
#     to the conversation history.
#     """
#     # Add user input to the conversation history
#     conversation_history.append({"role": "user", "content": user_input})

#     # Generate Wolfe's response using OpenAI's ChatCompletion API
#     response = openai.chat.completions.create(
#         model="gpt-4o",
#         max_tokens=100,
#         temperature=0.3,
#         messages=conversation_history
#     )

# # Extract Wolfe's response
#     ai_response = response["choices"][0]["message"]["content"]

#     # Add Wolfe's response to the conversation history
#     conversation_history.append({"role": "assistant", "content": ai_response})

#     # Return Wolfe's response
#     return ai_response
















# import os
# import openai

# # Retrieve the API key from the environment variable
# api_key = os.getenv('API_KEY')

# if api_key is None:
#     raise ValueError("No API key found. Please set the API_KEY environment variable.")

# # Set the API key for the OpenAI client
# openai.api_key = api_key

# # Example usage of the OpenAI API
# completion = openai.chat.completions.create(

# model="gpt-4o",
# max_tokens= 100,
# temperature=0.3,

# messages=[

# {"role": "system", "content": "You are wolfe, a compassionate and supportive AI therapist. Use the provided context to guide your response."
#                                 "The context contains scripts and techniques used by professional therapists. Your role is to validate"
#                           "the user's feelings, be conversational, ask thoughtful questions that can let you understand them and provide them responses that"
#                            "can help them feel comfortable with you, trust you, and open up more"
#                             "this relationship should be quite established before you begin to provide thoughtful guidance based on the context. Keep your response warm,"
#                              "empathetic, and non-judgmental."
#                              "without making assumptions or adding unnecessary context"
#                               "Keep your responses concise,"},

# {"role": "user", "content": "i will be taking a leave of absence from school today" }

# ]

# )

# print(completion.choices[0].message.content)



" "
#                                           
#  "focused, and relevant to the user's statement."











# ?Main Prompt
# import os
# import openai

# # Retrieve the API key from the environment variable
# api_key = os.getenv('API_KEY')

# if api_key is None:
#     raise ValueError("No API key found. Please set the API_KEY environment variable.")

# # Set the API key for the OpenAI client
# openai.api_key = api_key

# # Example usage of the OpenAI API
# completion = openai.chat.completions.create(

# model="gpt-4o",
# max_tokens= 100,
# temperature=0.3,

# messages=[

# {"role": "system", "content": "You are a helpful assistant."},

# {"role": "user", "content": "can you tell me about steve jobs experience with synchronicities."}

# ]

# )

# print(completion.choices[0].message.content)









































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

#     # Convert response to dictionary and extract the embedding
#     query_embedding = np.array([response.model_dump()["data"][0]["embedding"]])

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
#         model="gpt-4o",
#         temperature=0.3,
#         messages=[
#             {"role": "system", "content": "You are wolfe, a compassionate and supportive AI therapist. Use the provided context to guide your response. "
#                                           "The context contains scripts and techniques used by professional therapists. Your role is to validate "
#                                           "the user's feelings, be conversational, ask thoughtful questions that can let you understand them and provide them responses that"
#                                           "can help them feel comfortable with you, trust you, and open up more"
#                                             "this relationship should be quite established before you begin to provide thoughtful guidance based on the context. Keep your response warm,"
#                                           "empathetic, and non-judgmental."
#                                           "without making assumptions or adding unnecessary context"
#                                           "Keep your responses concise, "
#  "focused, and relevant to the user's statement."},
#             {"role": "user", "content": f"Context:\n{context}\n\n{user_query}"}
#         ],
#         max_tokens=100
#     )

#     # Convert response to dictionary and extract the message
#     return response.model_dump()["choices"][0]["message"]["content"]

# if __name__ == "__main__":
#     # Paths to required files
#     documents_path = "processed_documents.json"
#     index_path = "document_index.faiss"

#     # Load documents
#     with open(documents_path, "r", encoding="utf-8") as f:
#         documents = json.load(f)

#     # Load FAISS index
#     index = faiss.read_index(index_path)

#     # Create document map
#     doc_map = []
#     for doc_id, doc in enumerate(documents):
#         for chunk_id in range(len(doc["chunks"])):
#             doc_map.append((doc_id, chunk_id))

#     # User query
#     user_query = "I have lost interest in going outside"

#     # Query the FAISS index
#     results = query_faiss_index(user_query, index, documents, doc_map)

#     # Combine relevant chunks into context
#     context = "\n".join([res["chunk"] for res in results])

#     # Generate response using context
#     response = get_response_with_context(user_query, context)

#     # Output the response
#     print("Response from assistant:")
#     print(response)


# from openai import OpenAI
# import faiss
# import numpy as np
# import json
# import os

# # Instantiate the OpenAI client
# client = OpenAI(api_key=os.getenv("API_KEY"))

# # Initialize conversation history
# conversation_history = [
#     {"role": "system", "content": 
#      "You are Wolfe, a compassionate and supportive AI therapist. Use the provided context to guide your response. "
#      "The context contains scripts and techniques used by professional therapists. Your role is to validate "
#      "the user's feelings, be conversational, ask thoughtful questions that can let you understand them and provide responses that "
#      "help them feel comfortable with you, trust you, and open up more.  "
#      "This relationship should be quite established before you begin to provide thoughtful guidance based on the context. "
#      "Keep your responses warm, empathetic, and non-judgmental. Avoid making assumptions or adding unnecessary context, and ask thoughtful, open-ended questions only when needed."
#      "Keep your responses concise."
#      "do not make up anything that is not in the context"
#     }
# ]

# def query_faiss_index(user_query, index, documents, doc_map, top_k=3):
#     """
#     Query the FAISS index with the user query and retrieve the top-k relevant chunks.
#     """
#     response = client.embeddings.create(
#         model="text-embedding-ada-002",
#         input=user_query
#     )
#     query_embedding = np.array([response.model_dump()["data"][0]["embedding"]])

#     distances, indices = index.search(query_embedding, top_k)

#     results = []
#     for idx in indices[0]:
#         doc_id, chunk_id = doc_map[idx]
#         results.append({
#             "document": documents[doc_id]["name"],
#             "chunk": documents[doc_id]["chunks"][chunk_id]
#         })

#     return results

# def chat_with_wolfe(user_input, index, documents, doc_map):
#     """
#     Handles back-and-forth conversation with Wolfe by appending user input and AI responses
#     to the conversation history.
#     """
#     # Query the FAISS index
#     results = query_faiss_index(user_input, index, documents, doc_map)

#     # Combine relevant chunks into context
#     context = "\n".join([res["chunk"] for res in results])

#     # Add the user's message and context to the conversation history
#     conversation_history.append({"role": "user", "content": f"Context:\n{context}\n\n{user_input}"})

#     # Call the OpenAI API with the updated conversation history
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=conversation_history,
#         max_tokens=150,
#         temperature=0.1
#     )

#     # Extract Wolfe's response
#     ai_response = response.model_dump()["choices"][0]["message"]["content"]

#     # Add Wolfe's response to the conversation history
#     conversation_history.append({"role": "assistant", "content": ai_response})

#     # Return Wolfe's response
#     return ai_response

# if __name__ == "__main__":
#     # Paths to required files
#     documents_path = "processed_documents.json"
#     index_path = "document_index.faiss"

#     # Load documents
#     with open(documents_path, "r", encoding="utf-8") as f:
#         documents = json.load(f)

#     # Load FAISS index
#     index = faiss.read_index(index_path)

#     # Create document map
#     doc_map = []
#     for doc_id, doc in enumerate(documents):
#         for chunk_id in range(len(doc["chunks"])):
#             doc_map.append((doc_id, chunk_id))

#     print("Wolfe: Hi there! How are you feeling today?")

#     while True:
#         # Get user input
#         user_input = input("You: ")
#         if user_input.lower() in {"exit", "quit"}:
#             print("Wolfe: Take care of yourself. Goodbye!")
#             break

#         # Get Wolfe's response
#         ai_response = chat_with_wolfe(user_input, index, documents, doc_map)

#         # Print Wolfe's response
#         print(f"Wolfe: {ai_response}")

























































# import os
# import openai

# # Retrieve the API key from the environment variable

# api_key = os.getenv('API_KEY')

# if api_key is None:
# 	raise ValueError("No API key found. Please set the API_KEY environment variable.")

# # Set the API key for the OpenAI client
# openai.api_key = api_key

# # Example usage of the OpenAI API
# completion = openai.chat.completions.create(
#     model="gpt-4o",
# 	max_tokens=150,

# messages=[

# {"role": "system", "content": "You are a compassionate and supportive therapist. Your role is to listen attentively, "
#         "validate the user's feelings, and provide thoughtful and constructive guidance. "
#         "Maintain a calm, empathetic, and non-judgmental tone at all times."},

# {"role": "user", "content": "i find it hard to be happy on some days."}

# ]

# )

# print(completion.choices[0].message.content)




















# # import os

# # # Set your API key
# # os.environ["RAWAPIKEY"] = 'your_api_key_here'

# # # Retrieve the API key
# # api_key = os.getenv('API_KEY')

# # # Use the API key in your code
# # print(f"Your API key is: {key}");

# import os

# # Retrieve the API key
# api_key = os.getenv('API_KEY')

# if api_key is None:
#     raise ValueError("No API key found. Please set the API_KEY environment variable.")

# # Use the API key in your code
# print(f"Your API key is: {api_key}")

# ...rest of your code...

# # import openai

# # openai.api_key = api_key

# # completion = openai.ChatCompletion.create(

# # model="gpt-4o",

# # messages=[

# # {"role": "system", "content":

# # "You are Wolfe, a deeply empathetic and emotionally intelligent AI assistant. "

# # "Your purpose is to provide a safe space for the user to open up, validate their feelings, and gently guide them toward understanding and action. "

# # "Structure your response as follows:\n"

# # "1. Acknowledge and validate the user's emotions with warmth and empathy (e.g., 'It sounds like you’re feeling...').\n"

# # "2. Prompt them to share more by asking open-ended, supportive questions (e.g., 'What’s been going on? Let’s talk about it.').\n"

# # "3. As they open up, reassure them that their feelings are valid and normalize their experience.\n"

# # "4. Gradually guide the conversation toward actionable steps when the user feels ready.\n\n"

# # "Always maintain a calm, conversational tone. Avoid rushing into solutions or advice before the user feels comfortable sharing."

# # },

# # {"role": "user", "content": "I feel like shit even though its christmas."}

# # ]

# # )

# # print(completion.choices[0].message.content)

# import openai

# openai.api_key = api_key

# response = openai.ChatCompletion.create(
#     model="gpt-4o",
#     messages=[

# {"role": "system", "content":

# "You are Wolfe, a deeply empathetic and emotionally intelligent AI assistant. "

# "Your purpose is to provide a safe space for the user to open up, validate their feelings, and gently guide them toward understanding and action. "

# "Structure your response as follows:\n"

# "1. Acknowledge and validate the user's emotions with warmth and empathy (e.g., 'It sounds like you’re feeling...').\n"

# "2. Prompt them to share more by asking open-ended, supportive questions (e.g., 'What’s been going on? Let’s talk about it.').\n"

# "3. As they open up, reassure them that their feelings are valid and normalize their experience.\n"

# "4. Gradually guide the conversation toward actionable steps when the user feels ready.\n\n"

# "Always maintain a calm, conversational tone. Avoid rushing into solutions or advice before the user feels comfortable sharing."

# },

# {"role": "user", "content": "I feel like shit even though its christmas."}

# ]

# )

# print(response.choices[0].message['content'])









# import os
# import openai
# import faiss
# import numpy as np
# import json

# # Retrieve the API key from the environment variable
# api_key = os.getenv('API_KEY')

# if api_key is None:
#     raise ValueError("No API key found. Please set the API_KEY environment variable.")

# # Set the API key for the OpenAI client
# openai.api_key = api_key

# # Load documents and FAISS index
# def load_documents(file_path):
#     """
#     Load the processed documents from a JSON file.
#     """
#     with open(file_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# def load_faiss_index(index_path):
#     """
#     Load the FAISS index from a file.
#     """
#     index = faiss.read_index(index_path)
#     return index

# def query_faiss_index(user_query, index, documents, doc_map, top_k=3):
#     """
#     Query the FAISS index with the user query and retrieve the top-k relevant chunks.
#     """
#     # Generate embedding for the user query
#     response = openai.Embedding.create(
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

# # Example usage of the OpenAI API with context
# if __name__ == "__main__":
#     # Paths to required files
#     documents_path = "processed_documents.json"
#     index_path = "document_index.faiss"

#     # Load documents and FAISS index
#     documents = load_documents(documents_path)
#     index = load_faiss_index(index_path)

#     # Create document map
#     doc_map = []
#     for doc_id, doc in enumerate(documents):
#         for chunk_id in range(len(doc["chunks"])):
#             doc_map.append((doc_id, chunk_id))

#     # User query
#     user_query = "I find it hard to be happy on some days."

#     # Query the FAISS index
#     results = query_faiss_index(user_query, index, documents, doc_map)

#     # Combine relevant chunks into context
#     context = "\n".join([res["chunk"] for res in results])

#     # Generate a response using OpenAI's ChatCompletion API
#     completion = openai.ChatCompletion.create(
#         model="gpt-4o",
#         max_tokens=150,
#         messages=[
#             {"role": "system", "content": "You are a compassionate and supportive therapist. Your role is to listen attentively, "
#                                           "validate the user's feelings, and provide thoughtful and constructive guidance. "
#                                           "Maintain a calm, empathetic, and non-judgmental tone at all times."},
#             {"role": "user", "content": f"Context:\n{context}\n\n{user_query}"}
#         ]
#     )

#     # Print the response
#     print(completion.choices[0].message.content)






# import os
# import openai
# import faiss
# import numpy as np
# import json

# # Retrieve the API key from the environment variable
# api_key = os.getenv('API_KEY')

# if api_key is None:
#     raise ValueError("No API key found. Please set the API_KEY environment variable.")

# # Set the API key for the OpenAI client
# openai.api_key = api_key

# # Function to query the FAISS index
# def query_faiss_index(user_query, index, documents, doc_map, top_k=3):
#     """
#     Query the FAISS index with the user query and retrieve the top-k relevant chunks.
#     """
#     # Generate embedding for the user query
#     response = openai.Embedding.create(
#         model="text-embedding-ada-002",
#         input=user_query
#     )

#     # Extract the embedding from the response
#     query_embedding = np.array([response['data'][0]['embedding']])

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

# # Example usage of the OpenAI API with FAISS
# if __name__ == "__main__":
#     # Paths to required files
#     documents_path = "processed_documents.json"
#     index_path = "document_index.faiss"

#     # Load documents
#     with open(documents_path, "r", encoding="utf-8") as f:
#         documents = json.load(f)

#     # Load FAISS index
#     index = faiss.read_index(index_path)

#     # Create document map
#     doc_map = []
#     for doc_id, doc in enumerate(documents):
#         for chunk_id in range(len(doc["chunks"])):
#             doc_map.append((doc_id, chunk_id))

#     # User query
#     user_query = "I find it hard to be happy on some days."

#     # Query the FAISS index
#     results = query_faiss_index(user_query, index, documents, doc_map)

#     # Combine relevant chunks into context
#     context = "\n".join([res["chunk"] for res in results])

#     # Generate a response using OpenAI's ChatCompletion API
#     completion = openai.ChatCompletion.create(
#         model="gpt-4o",
#         max_tokens=150,
#         messages=[
#             {"role": "system", "content": "You are a compassionate and supportive therapist. Your role is to listen attentively, "
#                                           "validate the user's feelings, and provide thoughtful and constructive guidance. "
#                                           "Maintain a calm, empathetic, and non-judgmental tone at all times."},
#             {"role": "user", "content": f"Context:\n{context}\n\n{user_query}"}
#         ]
#     )

#     # Print the response
#     print(completion.choices[0].message.content)





# from openai import OpenAI
# import faiss
# import numpy as np
# import json
# import os




# # Initialize the OpenAI client
# client = OpenAI(
#     api_key=os.getenv("API_KEY")  # Ensure your environment variable is set
# )

# def query_faiss_index(user_query, index, documents, doc_map, top_k=3):
#     """
#     Query the FAISS index with the user query and retrieve the top-k relevant chunks.
#     """
#     # Generate embedding for the user query
#     response = client.embeddings.create(
#         model="text-embedding-ada-002",
#         input=user_query
#     )

#     # Convert response to dictionary and extract the embedding
#     query_embedding = np.array([response.model_dump()["data"][0]["embedding"]])

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
#         model="gpt-4o",
#         messages=[
#             {"role": "system", "content": "You are Wolfe, a deeply empathetic AI therapist. Your role is to listen attentively, validate the user's feelings, "
#                                             "and provide thoughtful guidance in a warm, conversational tone. Keep your responses concise and focus on connection, "
#                                              "not solutions. Use simple language, and let the user feel heard and supported."},
#             {"role": "user", "content": f"Context:\n{context}\n\n{user_query}"}
#         ],
#         max_tokens=150
#     )

#     # Convert response to dictionary and extract the message
#     return response.model_dump()["choices"][0]["message"]["content"]

# if __name__ == "__main__":
#     # Paths to required files
#     documents_path = "processed_documents.json"
#     index_path = "document_index.faiss"

#     # Load documents
#     with open(documents_path, "r", encoding="utf-8") as f:
#         documents = json.load(f)

#     # Load FAISS index
#     index = faiss.read_index(index_path)

#     # Create document map
#     doc_map = []
#     for doc_id, doc in enumerate(documents):
#         for chunk_id in range(len(doc["chunks"])):
#             doc_map.append((doc_id, chunk_id))

#     # User query
#     user_query = "I find it hard to be happy on some days."

#     # Query the FAISS index
#     results = query_faiss_index(user_query, index, documents, doc_map)

#     # Combine relevant chunks into context
#     context = "\n".join([res["chunk"] for res in results])

#     # Generate response using context
#     response = get_response_with_context(user_query, context)

#     # Output the response
#     print("Response from assistant:")
#     print(response)
