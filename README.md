# AI Legal Auditor ⚖️

**AI Legal Auditor** is a high-precision Retrieval-Augmented Generation (RAG) prototype designed to ingest, analyze, and risk-assess complex legal contracts. This solution was specifically developed as a prototype for the Fullstack AI Engineer recruitment process at AP Automated Ltd.

## 🚀 Project Overview
Legal documents are often dense and structurally complex. **AI Legal Auditor** solves the "needle-in-a-haystack" problem by extracting specific commercial clauses and providing an executive-level risk assessment with actionable redline suggestions.

### Key Objectives:
- **Automatic Clause Identification:** Precision extraction of IP, Price, Non-compete, Termination, and Governing Law.
- **Risk Assessment:** Dynamic scoring (1-10) based on vendor-friendly vs. customer-friendly language.
- **Contextual Integrity:** Leveraging Parent Document Retrieval to ensure legal nuance is never lost.

## 📊 Requirement Traceability Matrix
This project fulfills the core requirements of the January 2026 Home Programming Assignment:

| Assignment Requirement | Implementation & Evidence | Status |
| :--- | :--- | :---: |
| **Objective Extraction** | Successfully isolated IP, Price, Non-compete, Termination, and Governing Law. | ✅ Met |
| **Risk Assessment** | Developed a logic-based scoring system (1-10) with visual RAG status indicators. | ✅ Met |
| **Plain English Summary** | Transformed legalese into "Business Impact" statements for stakeholders. | ✅ Met |
| **Data Strategy (Task A)** | Implemented **Parent Document Retrieval** for high-precision citations (e.g., Section 2.1(b)). | ✅ Met |
| **AI Safety (Hallucination)** | Configured system to report "Silent/Not Found" rather than hallucinating missing clauses. | ✅ Met |
| **Bonus: Redline Suggestion** | Developed a generative layer to suggest balanced alternatives for High-Risk clauses. | 🌟 Exceeded |

## 🛠️ Technical Stack
- **Language:** Python 3.10+
- **LLM:** Gemini 3 Flash Preview (Optimized for long-context legal reasoning)
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Framework:** LangChain (ParentDocumentRetriever, RecursiveCharacterTextSplitter)

## 🧠 Statement of Contribution
My primary contribution to the RAG algorithm involves the custom integration of Parent Document Retrieval (PDR). Unlike standard RAG which retrieves isolated snippets, this architecture maps small "Child Chunks" (used for efficient searching) back to their "Parent" paragraphs. This ensures the LLM receives the complete surrounding context of a legal clause, drastically reducing the risk of misinterpretation.

## 📜 PDF Export 
Automatically generates formatted PDF reports with professional CSS styling for risk tables.

## 🛡️ Hallucination Monitoring
-To ensure reliability in a legal setting:
-Deterministic Sampling: Set temperature=0 to eliminate creative variance.
-Grounding: The system is strictly instructed to return an "I don't know" response if a clause is missing.
-Citations: Every audit identifies the exact section number where information was found.

## Prepared by: Syeda Neha Zafar



## 📋 Setup & Usage Instructions

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/AI-Legal-Auditor.git](https://github.com/yourusername/AI-Legal-Auditor.git)
cd AI-Legal-Auditor

2. Install dependencies
Bash

pip install -r requirements.txt
3. Set up environment
Create a .env file in the root directory and add your Google API Key:

GOOGLE_API_KEY=your_actual_key_here

4. Run the Pipeline
First, ingest the contracts into the vector store:

Run

python ingest.py

Then, execute the audit:

python main.py


