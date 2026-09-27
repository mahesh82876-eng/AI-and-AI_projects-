# Basic PDF RAG System

## 1. Project Overview

This project implements a **Retrieval-Augmented Generation (RAG) system** that allows an AI application to answer questions using information retrieved from a PDF document.

Instead of relying entirely on an LLM's existing knowledge, the system first searches the provided document for relevant information and then uses that information to generate an answer.

---

## 2. Objective

Build a foundational RAG pipeline that can:

* Load a PDF document
* Extract its text
* Convert the text into LangChain documents
* Split the document into smaller chunks
* Generate vector embeddings for the chunks
* Store the embeddings in ChromaDB
* Retrieve relevant chunks based on a user query
* Pass the retrieved information to an LLM
* Generate a context-aware answer

---

## 3. Technology Stack

| Component               | Technology                                            |
| ----------------------- | ----------------------------------------------------- |
| Programming Language    | Python                                                |
| RAG Framework           | LangChain                                             |
| PDF Processing          | PyMuPDF                                               |
| Document Representation | LangChain `Document`                                  |
| Text Splitting          | `RecursiveCharacterTextSplitter`                      |
| Embedding Model         | Hugging Face `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database         | ChromaDB                                              |
| LLM                     | To be added                                           |
| Environment             | Google Colab                                          |

---

# 4. RAG Architecture

```text
                    PDF DOCUMENT
                         │
                         ▼
                  PDF Text Extraction
                         │
                         ▼
                LangChain Documents
                         │
                         ▼
                    Chunking
                         │
                         ▼
                  Text Embeddings
                         │
                         ▼
                     ChromaDB
                         │
                         │
                  User Question
                         │
                         ▼
                Similarity Search
                         │
                         ▼
               Relevant PDF Chunks
                         │
                         ▼
                       LLM
                         │
                         ▼
                  Final Answer
```

---

# 5. Project Pipeline

## Step 1: Load the PDF

PyMuPDF is used to open the PDF and access its pages.

```python
import pymupdf

pdf = pymupdf.open("your_document.pdf")
```

---

## Step 2: Convert PDF Pages into LangChain Documents

Each PDF page is converted into a LangChain `Document`.

```python
from langchain_core.documents import Document

documents = []

for page_number, page in enumerate(pdf):

    text = page.get_text()

    if text.strip():

        documents.append(
            Document(
                page_content=text,
                metadata={"page": page_number + 1}
            )
        )
```

Each document contains:

```text
Document
├── page_content
│      └── Text extracted from PDF
│
└── metadata
       └── Page number
```

The page metadata allows us to know where retrieved information came from.

---

# 6. Step 3: Chunking

Large documents cannot always be processed efficiently as one large piece of text.

Therefore, we split the documents into smaller chunks.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)
```

### Current configuration

```text
Chunk size:     500 characters
Chunk overlap:   50 characters
```

The overlap helps preserve context between neighboring chunks.

---

# 7. Step 4: Generate Embeddings

We use the Hugging Face embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Implementation:

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

We can test the embedding:

```python
vector = embeddings.embed_query(
    chunks[0].page_content
)

print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])
```

The embedding model converts text into numerical vectors.

Conceptually:

```text
Text
 ↓
Embedding Model
 ↓
[0.21, -0.04, 0.73, ...]
```

These vectors allow us to compare the semantic similarity between text.

---

# 8. Step 5: Store Embeddings in ChromaDB

LangChain's Chroma integration is used as the vector store.

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings
)

vector_store.add_documents(chunks)
```

The vector store contains:

```text
Chunk
 +
Embedding
 +
Metadata
```

for the document.

---

# 9. Step 6: Retrieval

When a user asks a question, we search ChromaDB for the most relevant chunks.

Example:

```python
query = "What does the document say about power?"

results = vector_store.similarity_search(
    query,
    k=3
)
```

`k=3` means we retrieve the three most relevant chunks.

We can inspect the results:

```python
for i, result in enumerate(results):

    print(f"\n--- Result {i + 1} ---")
    print(result.page_content)
    print("Page:", result.metadata["page"])
```

The retrieval process is:

```text
User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
ChromaDB
      ↓
Top-k Relevant Chunks
```

---

# 10. Current Project Status

```text
PDF Loading              ✅
Text Extraction          ✅
LangChain Documents      ✅
Chunking                 ✅
Embeddings               ✅
ChromaDB                 ✅
Vector Retrieval         ✅
LLM Generation           ⏳
Final RAG Pipeline       ⏳
```

---

# 11. Important Concept

The current system separates **retrieval** from **generation**.

### Retrieval

Responsible for finding relevant information.

```text
Question
   ↓
Embedding
   ↓
ChromaDB
   ↓
Relevant chunks
```

### Generation

Responsible for producing the final natural-language answer.

```text
Question
   +
Retrieved chunks
   ↓
LLM
   ↓
Answer
```

The LLM is therefore **not responsible for searching the PDF**. ChromaDB performs the vector search, while the embedding model converts text and queries into vectors.

---

# 12. Final Goal

The completed system will follow:

```text
              USER
                │
                ▼
            QUESTION
                │
                ▼
         EMBEDDING MODEL
                │
                ▼
            CHROMADB
                │
                ▼
       RELEVANT PDF CHUNKS
                │
                ├──────────────┐
                │              │
                ▼              ▼
           USER QUERY      CONTEXT
                │              │
                └──────┬───────┘
                       ▼
                      LLM
                       │
                       ▼
                  FINAL ANSWER
```

This project represents the **foundational RAG architecture**. Once it is complete, it can be extended with techniques such as hybrid retrieval, reranking, query transformation, contextual retrieval, metadata filtering, evaluation, and production deployment.
