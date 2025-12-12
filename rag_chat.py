# rag_chat.py
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery, QueryType

# Load environment variables
load_dotenv()

AZURE_OPENAI_ENDPOINT = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_API_KEY = str(os.getenv("AZURE_OPENAI_API_KEY"))
GPT_MODEL_NAME = str(os.getenv("GPT_MODEL_NAME"))
EMBEDDING_MODEL_NAME = str(os.getenv("EMBEDDING_MODEL_NAME"))
SEARCH_ENDPOINT = str(os.getenv("SEARCH_ENDPOINT"))
SEARCH_API_KEY = str(os.getenv("SEARCH_API_KEY"))
SEARCH_INDEX_NAME = str(os.getenv("SEARCH_INDEX_NAME"))
OPENAI_API_VERSION = str(os.getenv("OPENAI_API_VERSION"))

# Initialize clients
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION
)
search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

def retrieve_context(query_text, top_k=5):
    """
    Performs a Hybrid Search (Vector + Text) on Azure AI Search.
    The query text is vectorized on-the-fly using text-embedding-ada-002
    via the Azure AI Search vectorizer configuration.
    """
    print(f"Retrieving context for query: '{query_text}'")
    
    # 1. Generate the embedding vector for the query
    embedding_response = openai_client.embeddings.create(
        input=query_text,
        model=EMBEDDING_MODEL_NAME
    )
    query_vector = embedding_response.data[0].embedding
    
    # 2. Prepare the Vector Query
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )

    # 2. Execute Hybrid Search
    results = search_client.search(
        search_text=query_text, # Keyword search
        vector_queries=[vector_query], # Vector search
        # Using Semantic Search for re-ranking the results (requires Basic tier or higher)
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="rag-semantic-config", 
        select=["content", "source_file"],
        top=top_k
    )

    retrieved_content = []
    
    for rank, result in enumerate(results):
        # A simple citation structure for the LLM prompt
        citation = f"[Citation {rank+1}: {result['source_file']}]"
        retrieved_content.append(f"{citation} {result['content']}")
        print(f"  - Found document (Score: {result['@search.score']}): {result['source_file']}")

    return "\n\n".join(retrieved_content)

def generate_answer(query_text, context):
    """Generates an answer using GPT-4o based on the retrieved context."""
    print("\nGenerating final answer with GPT-4o...")
    
    # System message to instruct the LLM
    system_message = (
        "You are an expert Q&A assistant. Your goal is to answer the user's question "
        "only using the provided CONTEXT. Do not use external knowledge. "
        "Cite the source documents using the citation format [Citation X: filename] provided in the CONTEXT."
    )
    
    # Full prompt for the LLM
    prompt = f"""
    {system_message}
    
    --- CONTEXT ---
    {context}
    ---
    
    QUESTION: {query_text}
    """

    response = openai_client.chat.completions.create(
        model=GPT_MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    
    return response.choices[0].message.content

def rag_chat_loop():
    """Main conversational loop."""
    print("--- Azure RAG Application (GPT-4o + Azure AI Search) ---")
    print("Type 'quit' or 'exit' to end the session.")
    
    while True:
        try:
            user_query = input("\nYour Question: ")
            if user_query.lower() in ['quit', 'exit']:
                break

            # 1. Retrieval Step (R)
            context = retrieve_context(user_query)
            
            if not context:
                print("--- Answer ---")
                print("Could not find any relevant documents in the index to answer your question.")
                continue

            # 2. Generation Step (G)
            answer = generate_answer(user_query, context)

            print("\n--- Answer ---")
            print(answer)
            
        except Exception as e:
            print(f"An error occurred: {e}")
            
if __name__ == "__main__":
    rag_chat_loop()