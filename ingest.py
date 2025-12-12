import os
import re
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration,
    VectorSearchProfile, AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    SemanticConfiguration, SemanticSearch, SemanticPrioritizedFields,
    SemanticField
)
from azure.search.documents import SearchClient

# Load environment variables
load_dotenv()

AZURE_OPENAI_ENDPOINT = str(os.getenv("AZURE_OPENAI_ENDPOINT"))
AZURE_OPENAI_API_KEY = str(os.getenv("AZURE_OPENAI_API_KEY"))
EMBEDDING_MODEL_NAME = str(os.getenv("EMBEDDING_MODEL_NAME"))
SEARCH_ENDPOINT = str(os.getenv("SEARCH_ENDPOINT"))
SEARCH_API_KEY = str(os.getenv("SEARCH_API_KEY"))
SEARCH_INDEX_NAME = str(os.getenv("SEARCH_INDEX_NAME"))
OPENAI_API_VERSION = str(os.getenv("OPENAI_API_VERSION"))

MAX_CHUNK_SIZE = 4000

# Initialize clients
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION
)
search_index_client = SearchIndexClient(
    endpoint=SEARCH_ENDPOINT,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

def create_search_index():
    """Creates a new Azure AI Search index with Vector Search and Semantic Search configuration."""
    print(f"Creating search index '{SEARCH_INDEX_NAME}'...")

    # 1. Define the Vectorizer (for text-embedding-ada-002)
    vectorizer = AzureOpenAIVectorizer(
        vectorizer_name="azure-openai-vectorizer",
        #kind="azureOpenAI",
        parameters=AzureOpenAIVectorizerParameters(
            resource_url=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=EMBEDDING_MODEL_NAME,
            model_name="text-embedding-ada-002"
        )
    )


    # 2. Define the Vector Search Profile (using HNSW algorithm)
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw-config")],
        profiles=[VectorSearchProfile(
            name="rag-vector-profile",
            algorithm_configuration_name="hnsw-config",
            vectorizer="azure-openai-vectorizer"
        )],
        vectorizers=[vectorizer]
    )
    
    # 3. Define Semantic Search Configuration
    semantic_search = SemanticSearch(
        configurations=[SemanticConfiguration(
            name="rag-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
                title_field=SemanticField(field_name="title")
            )
        )]
    )

    # 4. Define the Fields
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchField(name="title", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchField(name="content", type=SearchFieldDataType.String, searchable=True, retrievable=True),
        SearchField(name="source_file", type=SearchFieldDataType.String, filterable=True, retrievable=True),
        
        # FIX 3: Ensure 'dimensions' and 'vector_search_profile' are explicitly set for the vector field
        SearchField(
            name="content_vector", 
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
            searchable=True, 
            vector_search_profile_name="rag-vector-profile",
            vector_search_dimensions=1536 # text-embedding-ada-002 dimension is 1536
        )
    ]
    
    # 5. Assemble the Index
    index = SearchIndex(
        name=SEARCH_INDEX_NAME,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search
    )

    # Delete existing index if it exists, then create the new one
    try:
        search_index_client.delete_index(SEARCH_INDEX_NAME)
        print("Existing index deleted.")
    except Exception as e:
        if "not found" not in str(e):
             print(f"Could not delete index: {e}")
             
    search_index_client.create_index(index)
    print("Search index created successfully.")

def get_text_chunks(file_path):
    """Parses a PDF and returns a list of text chunks."""
    text_chunks = []
    
    try:
        reader = PdfReader(file_path)
        print(f"  - Reading {len(reader.pages)} pages from {os.path.basename(file_path)}")
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        # Simple chunking: split text by a large separator and ensure chunk size
        # In a production app, use LangChain or other advanced chunking (recursive, semantic)
        # to ensure meaningful chunks, as the user assignment specifies.
        MAX_CHUNK_SIZE = 4000
        
        # Split by double newline/newline/dot (simple sentence split)
        delimiters = ["\n\n", "\n", ". "]
        for delimiter in delimiters:
            if len(full_text) > MAX_CHUNK_SIZE:
                parts = full_text.split(delimiter)
                # Re-assemble parts until they are too big
                current_chunk = ""
                for part in parts:
                    if len(current_chunk) + len(part) < MAX_CHUNK_SIZE:
                        current_chunk += part + delimiter.strip()
                    else:
                        if current_chunk:
                            text_chunks.append(current_chunk.strip())
                        current_chunk = part + delimiter.strip()
                if current_chunk:
                    text_chunks.append(current_chunk.strip())
                # Replace the full_text with the final list of chunks for further processing if needed
                full_text = "\n\n".join(text_chunks)
                if len(full_text) < MAX_CHUNK_SIZE:
                    break
                else:
                    text_chunks = [] # Reset for next delimiter
            else:
                text_chunks = [full_text]
                break

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        
    return text_chunks


# ---------------------------------------------------------
# Step 1: Extract clean section-wise text from PDF
# ---------------------------------------------------------
def extract_sections(file_path):
    reader = PdfReader(file_path)

    pages = []
    for page in reader.pages:
        txt = page.extract_text()
        if txt:
            pages.append(txt)

    full_text = "\n".join(pages)

    # ---- Clean common PDF formatting issues ----
    full_text = re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2", full_text)     # de-hyphenation
    full_text = re.sub(r"\n+", "\n", full_text)                    # collapse newlines
    full_text = full_text.replace("ﬁ", "fi").replace("ﬂ", "fl")    # ligatures
    full_text = re.sub(r"[ \t]+", " ", full_text)                  # spaces

    # ---- Strong regex for academic section headers ----
    section_header_regex = re.compile(
        r"(?P<header>"
        r"Abstract|"
        r"[0-9]+(?:\.[0-9]+)*\s+[A-Z][A-Za-z0-9 ,;\-\(\)]+"
        r")"
        r"\s*\n",
        re.MULTILINE,
    )

    matches = list(section_header_regex.finditer(full_text))
    if not matches:
        return {"FULL_DOCUMENT": full_text}

    sections = {}

    for i, match in enumerate(matches):
        header = match.group("header").strip()
        start = match.end()

        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(full_text)

        body = full_text[start:end].strip()

        # remove accidental page numbers
        body = re.sub(r"\n?\s*\d+\s*\n", " ", body)

        sections[header] = body

    return sections


# ---------------------------------------------------------
# Step 2: Chunk text while keeping words intact
# ---------------------------------------------------------
def chunk_text(text, max_size=MAX_CHUNK_SIZE):
    words = text.split(" ")
    chunks = []
    current = ""

    for w in words:
        if len(current) + len(w) + 1 <= max_size:
            current += w + " "
        else:
            chunks.append(current.strip())
            current = w + " "

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---------------------------------------------------------
# Step 3: Produce final result as list[str]
# ---------------------------------------------------------
def extract_and_chunk_sections(file_path, max_size=MAX_CHUNK_SIZE):
    """
    Returns list[str]:
        [
            "Abstract - chunk1",
            "Abstract - chunk2",
            "1 Introduction - chunk1",
            "1 Introduction - chunk2",
            ...
        ]
    """
    sections = extract_sections(file_path)
    final_list = []

    for section_title, section_text in sections.items():
        chunks = chunk_text(section_text, max_size=max_size)

        for chunk in chunks:
            final_list.append(f"{section_title} - {chunk}")

    return final_list



def get_embeddings(texts):
    """Generates embeddings using Azure text-embedding-ada-002."""
    print(f"  - Generating {len(texts)} embeddings...")
    response = openai_client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL_NAME
    )
    return [item.embedding for item in response.data]

def index_documents():
    """Reads PDF files, chunks them, generates embeddings, and uploads to Azure Search."""
    print("Starting document indexing process...")
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX_NAME,
        credential=azure.core.credentials.AzureKeyCredential(SEARCH_API_KEY)
    )

    documents_to_upload = []
    document_id_counter = 0

    for filename in os.listdir("PDF_Files"):
        if filename.endswith(".pdf"):
            file_path = os.path.join("PDF_Files", filename)
            
            print(f"Processing file: {filename}")
            #chunks = get_text_chunks(file_path)
            chunks = extract_and_chunk_sections(file_path)
            embeddings = get_embeddings(chunks)
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                document_id_counter += 1
                doc = {
                    # Azure Search 'id' field must be a string and unique
                    "id": str(document_id_counter),
                    "title": filename,
                    "source_file": filename,
                    "content": chunk,
                    "content_vector": embedding
                }
                documents_to_upload.append(doc)

    if documents_to_upload:
        print(f"\nUploading {len(documents_to_upload)} documents to Azure AI Search...")
        result = search_client.upload_documents(documents=documents_to_upload)
        # Check if any documents failed to upload
        failed_count = sum(1 for r in result if not r.succeeded)
        print(f"Upload complete. Succeeded: {len(result) - failed_count}, Failed: {failed_count}")
    else:
        print("No documents were processed or uploaded.")

if __name__ == "__main__":
    import azure.core.credentials
    create_search_index()
    index_documents()