import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever

load_dotenv()

# Configuration
Docs_Folder= "./documents"
DB_Path="./vector_store"
Store_Path = "./vector_store/parent_chunks"

def clean_vector_store():
    # Clears existing DB to prevent duplicates data during testing
    if os.path.exists(DB_Path):
        shutil.rmtree(DB_Path)
    os.makedirs(DB_Path, exist_ok=True)
    os.makedirs(Store_Path, exist_ok=True)
    print ("Cleared Old Vector store.")
    
def ingest_data():
    # 1.Load PDFs
    docs=[]
    if not os.path.exists(Docs_Folder):
        os.makedirs(Docs_Folder)
        print (f"documents folder created. Please put your PDF contracts inside '{Docs_Folder}' and rerun.")
        return 
    print ("Loading Documents...")
    
    for file in os.listdir(Docs_Folder):
        if file.lower().endswith(".pdf"):
            file_path= os.path.join(Docs_Folder, file)
            print(f"   - Processing: {file}")
            loader = PyPDFLoader(file_path)
            #Basic cleaning: Merge pages to aoid header/footer breaks in middle of sentences
            docs.extend(loader.load())
    
    if not docs:
        print (" No PDF files found in documents.")
        return
    
    # 2. Strategy: Parent Document Retrival
    # we use two splitters:
    #     Child Splitters: Small Chunks (Acuurate Search)
    #     Paent Splitters: Large Chunks (full context for LLM) 
                
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size= 2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    # 3. Vector Database
    print("Initializing Local Embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore= Chroma(
        collection_name="gemini_legal_audit",
        embedding_function=embedding_model ,
        persist_directory= DB_Path
    )
    
    # 4. Storage for Parent Chunks (the "Long Context" Requirement)
    fs = LocalFileStore(Store_Path)
    store = create_kv_docstore(fs)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter= parent_splitter,
    )
    
    print("Indexing documents (generating Parent-Child chunks)...")
    retriever.add_documents(docs)
    print("Ingestion Complete. Data is vectorized ans stored. Ready for auditing.")
    
if __name__ == "__main__":
    clean_vector_store()
    ingest_data()
    