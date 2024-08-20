# Medical Chatbot using Llama3.1 model.


## Steps to create a quantised model to use locally for a chatbot
Performed Quantization of Llama3.1 8B Instruct model from huggingface to 4 bit quantised version using llama.cpp
STEPS USED:

1. Downloading the Meta-Llama-3.1–8B-Instruct Model from huggingface
```
  huggingface-cli login
  huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir Meta-Llama-3.1-8B-Instruct
```

2. Quantisation process using llama.cpp
 ```
    git clone https://github.com/ggerganov/llama.cpp.git
    python3 -m pip install -r llama.cpp/requirements.txt
    make -C llama.cpp -j4
```

3. Using llama-quantize to generate Meta-Llama-3.1-8B-Instruct--q4_0.bin
 ```  
   python3 llama.cpp/convert_hf_to_gguf.py Meta-Llama-3.1-8B-Instruct
  ./llama.cpp/llama-quantize Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-F16.gguf Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct--q4_0.bin q4_0
```
## Using the quantised model for chatbot

## Process & Tech Stack ##
1. Loading the PDF file containing medical data using **PyPdfLoader**
2. Perform chunking and embedding the storing the vector representation of the chunks in vector database using **RecursiveCharacterTextSplitter** from Langchain
3. Embedding model from huggingface **sentence-transformers/all-MiniLM-L6-v2** is used to create the embeddings.
4. **Pinecone** is used as a vector DB to store the indexes created for chunks.
5. **LlamaCpp** from the langchain community is used to load saved quantised model and later for inference.
6. **Langchain** is used to create a prompt template and Retrieval QA to generate response.
7. **Flask API** is used to create the frontend.

## Final outcome ##
<img width="824" alt="Screenshot 2024-08-20 at 6 22 34 PM" src="https://github.com/user-attachments/assets/983ee66d-22b4-485b-9b6c-ef5aeffa658d">
<img width="829" alt="Screenshot 2024-08-20 at 6 29 22 PM" src="https://github.com/user-attachments/assets/c4b7bf01-b9ee-4834-a05b-b92191ab42dc">


