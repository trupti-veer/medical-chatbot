from src.helper import load_pdf, text_split, download_huggingface_embedding
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

#Step 1: Load pdf data
extracted_data = load_pdf("data/")

#Step 2: Split large pdf in small chunks.
text_chunks = text_split(extracted_data)

#Step 3: Download huggingface embedding model
embeddings = download_huggingface_embedding()

#Step 4: Initialise Pine cone and store chunks into Vector DB using the index created.
index_name = "med-bot"
docs = PineconeVectorStore.from_texts([t.page_content for t in text_chunks], embeddings, index_name=index_name)



