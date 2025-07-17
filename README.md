# RAG Pipeline for ICD-10 Code Retrieval

## Overview
This project implements a Retrieval-Augmented Generation (RAG) pipeline to retrieve ICD-10 codes from a PDF document based on user queries. The pipeline processes an ICD-10 PDF, splits it into chunks, generates embeddings, stores them in a FAISS vector store, and uses a lightweight language model (LLM) to generate answers with source document references.

## Tools and Models Used
- **Programming Language**: Python 3
- **Libraries**:
  - `langchain`: For document loading (`PyPDFLoader`), text splitting (`RecursiveCharacterTextSplitter`), embeddings (`HuggingFaceEmbeddings`), vector store (`FAISS`), and RAG pipeline (`RetrievalQA`).
  - `transformers`: For loading the LLM and tokenizer (`AutoTokenizer`, `AutoModelForCausalLM`, `pipeline`).
  - `faiss-cpu`: For efficient vector storage and similarity search.
  - `torch`: For dynamic CPU/GPU support with model inference.
  - `logging`: For runtime logging and debugging.
- **Models**:
  - **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) for lightweight, efficient embeddings suitable for text retrieval.
  - **LLM**: `facebook/opt-350m` (HuggingFace) for text generation, chosen for its small size and compatibility with resource-constrained environments.
- **Development Environment**: Local Python environment, compatible with Jupyter Notebook or `.py` file execution.

## Use of AI Tools
- **ChatGPT/Copilot**: Not used directly for coding or debugging. The implementation was developed manually using prior knowledge, LangChain documentation, HuggingFace documentation, and standard Python practices.
- **General Guidance**: The pipeline structure was informed by common RAG patterns from online tutorials and community resources, which may indirectly reflect AI tool influences in the broader ecosystem. No specific AI-generated code was copied or used.
- **Manual Implementation**: All code was written and debugged manually, with stack traces and official documentation as primary debugging tools.

## Design Decisions and Assumptions
- **Document Loading**: Used `PyPDFLoader` to extract text from the ICD-10 PDF (`9241544228_eng.pdf`), assuming the file is accessible in the working directory and contains structured ICD-10 code data.
- **Text Splitting**: Employed `RecursiveCharacterTextSplitter` with a chunk size of 500 tokens and 50-token overlap to balance context retention and memory efficiency, ensuring chunks are suitable for embedding and retrieval.
- **Embedding Model**: Selected `all-MiniLM-L6-v2` for its fast performance and low resource requirements, ideal for standard hardware.
- **Vector Store**: Utilized FAISS for efficient similarity search, with Maximal Marginal Relevance (MMR) to enhance diversity in retrieved documents.
- **LLM Choice**: Chose `facebook/opt-350m` for its lightweight footprint, enabling fast inference on CPU or single GPU, suitable for the 4-hour development constraint.
- **Retrieval Configuration**: Configured the retriever to fetch the top 5 documents (`k=5`) using MMR (`search_type="mmr"`) to avoid redundant results and improve relevance.
- **Error Handling**: Implemented comprehensive logging and try-catch blocks to handle errors robustly and aid debugging.
- **Assumptions**:
  - The provided PDF contains structured ICD-10 data that can be queried effectively.
  - The sample query ("Recurrent depressive disorder, currently in remission") is representative of typical user inputs.
  - Hardware supports PyTorch with CPU or CUDA (dynamically detected via `device_map="auto"`).
  - The 4-hour time limit prioritizes core functionality over optional features.

## Implemented Features
- **End-to-End Pipeline**:
  - Loads and processes the ICD-10 PDF using `PyPDFLoader`.
  - Splits text into 500-token chunks with 50-token overlap using `RecursiveCharacterTextSplitter`.
  - Generates embeddings with `all-MiniLM-L6-v2` and stores them in a FAISS vector store.
  - Integrates `facebook/opt-350m` for answer generation via `HuggingFacePipeline`.
  - Uses `RetrievalQA` with MMR to retrieve and answer queries, returning both the answer and source documents with metadata (e.g., page numbers).
- **Code Quality**:
  - Modular structure with distinct functions (`load_document`, `split_documents`, `create_vector_store`, `setup_llm`, `create_rag_pipeline`, `query_rag`).
  - Clear comments and logging for readability and debugging.
  - Robust error handling with try-catch blocks and detailed error messages.
- **Technical Design**:
  - Lightweight models chosen for resource efficiency.
  - MMR used to improve retrieval diversity.
  - Dynamic device support (CPU/GPU) via PyTorch.
- **Time Management**:
  - Completed core pipeline within ~3.5 hours, adhering to the 4-hour limit.
  - Prioritized functional MVP over optional features.

## Non-Implemented Features and Reasons
- **Gradio or Streamlit Interface**:
  - **Reason**: Skipped due to the 4-hour time constraint. Developing a UI would require 1-2 hours for setup, design, and testing, which would risk incomplete core functionality.
  - **Impact**: The pipeline is command-line based, functional but less accessible to non-technical users.
- **Dynamic Document Uploads**:
  - **Reason**: Omitted to focus on core pipeline with a single static PDF. Dynamic uploads would require additional file handling logic and possibly a UI, adding complexity beyond the time limit.
  - **Impact**: Limits flexibility to process multiple or user-uploaded documents but simplifies the MVP scope.
- **Basic Caching for Repeat Queries**:
  - **Reason**: Not implemented due to time constraints. Caching (e.g., using a dictionary or Redis) would require extra logic and testing, estimated at 30-60 minutes.
  - **Impact**: Repeat queries are reprocessed, potentially increasing runtime.
- **Advanced Relevance Reranking**:
  - **Reason**: While MMR was used in the retriever, custom reranking (e.g., using a scoring function or external model) was skipped due to time limits and complexity.
  - **Impact**: MMR provides diversity, but advanced reranking could further improve relevance for complex queries.
- **Why Skipped**:
  - **Time Constraint**: The 4-hour limit necessitated prioritizing the core pipeline (document processing, retrieval, answering) over optional features.
  - **Resource Constraints**: Lightweight models were chosen for compatibility with standard hardware, limiting exploration of larger models or GPU-optimized FAISS.
  - **MVP Focus**: The task emphasized a clean, working MVP without overengineering, so optional features were deprioritized.
  - **Static PDF Assumption**: Assuming a single PDF simplified the scope within the time limit.

## Limitations
- **Time Constraints**:
  - Core pipeline completed, but optional features (UI, dynamic uploads, caching, advanced reranking) were skipped to meet the 4-hour limit.
  - Estimated time for optional features: 2-3 additional hours.
- **Resource Constraints**:
  - Lightweight models (`all-MiniLM-L6-v2`, `opt-350m`) ensure compatibility but may limit answer quality compared to larger models (e.g., LLaMA 7B).
  - FAISS is CPU-based; GPU-accelerated FAISS could improve performance but was not explored due to setup complexity.
- **LLM Performance**: `opt-350m` may produce less precise answers for complex medical queries compared to larger models.
- **Single PDF Scope**: The pipeline assumes a single ICD-10 PDF, limiting flexibility for dynamic document processing.

## Sample Output
**Query**: "Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission"

**Answer** (example, actual output depends on PDF content):
Query: Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission
Answer: Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.        

Depressive
- disorder  {see  Disorder, depressive)
- episode  {see  Episode, depressive)
332

F99 UNSPECIFIED MENTAL DISORDER
EH Mental disorder, not otherwise specified
Non-recommended residual category, when no other code from
F00-F98 can be used.
291

preferably with an epidemiological basis, that would strengthen the case for
these inclusions as disorders clinically distinguishable from others already in-
the classification have failed, so they have not been separately classified. Descrip-
tions of these disorders currently available in the literature suggest that they
may be regarded as local variants of anxiety, depression, somatoform disorder,
or adjustment disorder; the nearest equivalent code should therefore be used

depressive type, and for a recurrent disorder in which the majority
of episodes are schizoaffective, depressive type.
Includes: schizoaffective psychosis, depressive type
schizophreniform psychosis, depressive type
F25.2 Schizoaffective disorder, mixed type
Disorders in which symptoms of schizophrenia (F20. - ) coexist
with those of a mixed bipolar affective disorder (F31.6) should be
coded here.
Includes: cyclic schizophrenia
mixed schizophrenic and affective psychosis

Diagnostic guidelines
For a definite diagnosis:
(a) the criteria for recurrent depressive disorder (F33. -) should be
fulfilled, and the current episode should fulfil the criteria for
depressive episode, moderate severity (F32.1); and
(b) at least two episodes should have lasted a minimum of 2 weeks
and should have been separated by several months without
significant mood disturbance.
Otherwise the diagnosis should be other recurrent mood [affective]
disorder (F38.1).

Question: Give me the correct coded classification for the following diagnosis: Recurrent depressive disorder, currently in remission
Helpful Answer:

Depressive
- disorder  {see  Disorder, depressive)
- episode  {see  Episode, depressive)
332

F99 UNSPECIFIED MENTAL DISORDER
EH Mental disorder, not otherwise specified
Non-recommended residual category, when no other code from F00-F98 can be used.
291

depressive type, and for a recurrent disorder in which the majority
of episodes are schizoaffective, depressive type.
Includes: schizoaffective psychosis, depressive type
disorders in which symptoms of schizophrenia (F20. - ) coexist
with those of a mixed bipolar affective disorder (F31.6) should be
coded here.
Includes: cyclic schizophrenia
mixed schizophrenic and affective psychosis

Diagnostic guidelines
For a definite diagnosis:
(a) the criteria for recurrent depressive disorder (F33. -) should be
fulfilled, and the

Source Documents:

Document 1:
Page: 344
Content: Depressive
- disorder  {see  Disorder, depressive)
- episode  {see  Episode, depressive)
332...

Document 2:
Page: 303
Content: F99 UNSPECIFIED MENTAL DISORDER
EH Mental disorder, not otherwise specified
Non-recommended residual category, when no other code from
F00-F98 can be used.
291...

Document 3:
Page: 28
Content: preferably with an epidemiological basis, that would strengthen the case for
these inclusions as disorders clinically distinguishable from others already in-
the classification have failed, so they ha...

Document 4:
Page: 120
Content: depressive type, and for a recurrent disorder in which the majority
of episodes are schizoaffective, depressive type.
Includes: schizoaffective psychosis, depressive type
schizophreniform psychosis, d...

Document 5:
Page: 138
Content: Diagnostic guidelines
For a definite diagnosis:
(a) the criteria for recurrent depressive disorder (F33. -) should be
fulfilled, and the current episode should fulfil the criteria for
depressive episo...
