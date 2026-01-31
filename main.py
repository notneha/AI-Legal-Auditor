import os
import re
import datetime
import json
# 1. Environment & Offline Setup (Must stay at top)
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Using standardized storage and retriever imports
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever

# Setup
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("Error: GOOGLE_API_KEY not found in .env file!")

# 2. Initialize the LLM (Gemini 3 Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

# 3. Load the Local Vector Store
print("Loading local embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="gemini_legal_audit",
    embedding_function=embeddings,
    persist_directory="./vector_store"
)

# 4. Connect to Parent Store & Initialize Retriever
fs = LocalFileStore("./vector_store/parent_chunks")
store = create_kv_docstore(fs)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

# 5. Define the Audit Prompt
template = """
You are a senior Legal Auditor. Provide a clean, executive-style summary for a business client.

INSTRUCTIONS:
- Start directly with a 'RISK SCORECARD' section using a scale of 1-10.
- Use emojis for Risk Levels: 🔴 (High/Deal-breaker), 🟡 (Medium), 🟢 (Low/Informational).
- Output a Markdown Table with: Clause Category, Risk Level, Identified Risk, Business Impact, and Suggested Redline.
- DO NOT include conversational intros, pleasantries, or JSON formatting in your response.

Analyze these five areas:
1. IP Ownership Assignment
2. Price Restrictions
3. Non-compete, Exclusivity, and No-solicit
4. Termination for Convenience
5. Governing Law

CONTEXT:
{context}

REQUEST:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def final_clean_text(text):
    """
    Consolidated cleaning function to handle both JSON wrappers 
    and AI conversational filler.
    """
    # If the response is wrapped in Gemini's list/dict structure
    if text.startswith("[{'") or text.startswith("{'"):
        match = re.search(r"'text':\s*'(.*?)'(?:,\s*'extras'|})", text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # Handle escaped newlines
    text = text.replace("\\n", "\n")
    
    # Remove technical metadata block ('extras')
    text = re.sub(r"'extras':\s*\{.*\}", "", text, flags=re.DOTALL)
    
    # Remove markdown code fence garbage
    text = re.sub(r"```[a-z]*\n?", "", text)
    
    # Remove AI chatter at start
    text = re.sub(r"^(Here is|Based on|I have analyzed|### Legal Audit Report).*\n", "", text, flags=re.IGNORECASE)
    
    return text.strip()

# The RAG Chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# 6. Execution Block
if __name__ == "__main__":
    print("Starting Professional Legal Audit Analysis...")
    try:
        query = "Extract and analyze the 5 key commercial clauses and assign risk scores."
        
        # Determine source filename dynamically
        source_docs = retriever.invoke(query)
        try:
            raw_path = source_docs[0].metadata.get('source', 'Contract')
            base_filename = os.path.splitext(os.path.basename(raw_path))[0]
        except (IndexError, KeyError):
            base_filename = "Contract"

        # Create unique timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"Audit_Report_{base_filename}_{timestamp}.md"

        # Run the AI Audit
        response = chain.invoke(query)
        
        # Clean the content thoroughly
        raw_content = str(response.content) if hasattr(response, 'content') else str(response)
        final_content = final_clean_text(raw_content)
        
        # Build Professional Header
        report_header = (
            f"# EXECUTIVE LEGAL AUDIT: {base_filename.replace('_', ' ').replace('-', ' ').upper()}\n"
            f"**Analyzed on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Model:** Gemini 3 Flash Preview\n"
            f"{'='*50}\n\n"
        )
        full_report = report_header + final_content
        
        # Print to console
        print("\n" + full_report)
        
        # SAVE TO FILE
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(full_report)
            
        print(f"\n Success! Professional audit saved to: {output_filename}")
        
    except Exception as e:
        print(f" An error occurred during the audit: {e}")