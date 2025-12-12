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

# ---- MEMORY CONFIG ----
conversation_history = []  # Stores last 4 turns
MAX_HISTORY = 4

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
    Hybrid Search (Vector + Text) on Azure AI Search.
    """
    print(f"Retrieving context for query: '{query_text}'")

    # 1. Vectorize the query using embeddings
    embedding_response = openai_client.embeddings.create(
        input=query_text,
        model=EMBEDDING_MODEL_NAME
    )
    query_vector = embedding_response.data[0].embedding

    # 2. Vector query
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="content_vector"
    )

    # 3. Execute Hybrid + Semantic search
    results = search_client.search(
        search_text=query_text,
        vector_queries=[vector_query],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name="rag-semantic-config",
        select=["content", "source_file"],
        top=top_k
    )

    retrieved_content = []

    for rank, result in enumerate(results):
        citation = f"[Citation {rank+1}: {result['source_file']}]"
        retrieved_content.append(f"{citation} {result['content']}")
        print(f"  - Found document (Score: {result['@search.score']}): {result['source_file']}")

    return "\n\n".join(retrieved_content)


def generate_answer(query_text, context):
    """
    Generate answer using GPT model + conversation memory.
    """
    print("\nGenerating final answer with GPT model...")

    # Build memory text
    memory_text = ""
    for turn in conversation_history:
        memory_text += f"USER: {turn['user']}\nASSISTANT: {turn['assistant']}\n\n"

    system_message = (
        "You are an expert RAG assistant. "
        "Use ONLY the provided CONTEXT to answer the question. "
        "Use MEMORY only for conversational coherence, but never for facts. "
        "Cite documents using the format [Citation X: filename]."
    )

    # Combined prompt
    prompt = f"""
    {system_message}

    --- MEMORY (Last {MAX_HISTORY} Turns) ---
    {memory_text if memory_text else 'No previous conversation.'}

    --- CONTEXT ---
    {context}
    ---

    QUESTION: {query_text}
    """

    response = openai_client.chat.completions.create(
        model=GPT_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    answer = response.choices[0].message.content
    return answer


def rag_chat_loop():
    """
    Main RAG chat loop with memory.
    """
    print("--- Azure RAG Application (GPT + Azure AI Search) ---")
    print("Type 'quit' or 'exit' to end the session.")

    while True:
        try:
            user_query = input("\nYour Question: ")

            if user_query.lower() in ["quit", "exit"]:
                break

            # 1. Retrieval
            context = retrieve_context(user_query)

            if not context:
                print("--- Answer ---")
                print("No relevant documents found for this question.")
                continue

            # 2. Generation
            answer = generate_answer(user_query, context)

            print("\n--- Answer ---")
            print(answer)

            # 3. Update memory
            conversation_history.append({
                "user": user_query,
                "assistant": answer
            })

            # Keep only last 4 turns
            if len(conversation_history) > MAX_HISTORY:
                conversation_history.pop(0)

        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    rag_chat_loop()
