import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()

var1 = os.getenv('EMBEDDINGS_DEPLOYMENT')
print(var1)

def data_extraction(filename):
    reader = PdfReader(filename)
    number_of_pages = len(reader.pages)
    text = ""
    for page_number in range(number_of_pages):
        page = reader.pages[page_number]
        text += page.extract_text()
    print(text)
    return "extracted_data"

def data_preprocessing_and_chunking(data):
    # Your data preprocessing logic here
    preprocessed_data = data.lower()  # Example preprocessing step
    return "embeddings"

def store_embeddings(embeddings):
    # Your logic to store embeddings here
    pass

pdf_folder = os.path.join(os.getcwd(), "PDF_Files")
files_list = []
for filename in os.listdir(pdf_folder):
    file_path = os.path.join(pdf_folder, filename)
    if os.path.isfile(file_path):
        files_list.append(file_path)

for file in files_list:
    # Your data extraction logic here
    extracted_data = data_extraction(file)

    # Your data preprocessing and chunking logic here
    embeddings = data_preprocessing_and_chunking(extracted_data)

    # Your embedding creation logic here
    store_embeddings(embeddings)