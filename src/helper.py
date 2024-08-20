from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

#Extract data from PDF file.
def load_pdf(data):
    loader = DirectoryLoader(data, 
                    glob="*.pdf", 
                    loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


#Create text chunking 
def text_split(extracted_data):
    splitted_text = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = splitted_text.split_documents(extracted_data)
    return text_chunks


#Download embedding model for the same.
def download_huggingface_embedding():
    embedddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embedddings