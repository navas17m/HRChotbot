"""
RAG Pipeline — LangChain + FAISS + HuggingFace Embeddings
Uses direct LCEL calls — no langchain.chains helpers required.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---- Paths ---------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
DATA_PATH  = str(BASE_DIR / "data")
FAISS_PATH = str(BASE_DIR / "faiss_index")

# ---- Embedding model -----------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RAGPipeline:
    """
    Full RAG pipeline:
      1. Load & chunk HR policy documents
      2. Embed with all-MiniLM-L6-v2 → store in FAISS
      3. Direct retriever + LLM call with chat history
    """

    def __init__(self):
        logger.info("Initialising RAG Pipeline …")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore = None
        self.retriever   = None
        self.llm         = None
        self._initialise()

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def _initialise(self):
        faiss_exists = (
            os.path.isdir(FAISS_PATH)
            and os.path.exists(os.path.join(FAISS_PATH, "index.faiss"))
        )
        if faiss_exists:
            logger.info("Loading existing FAISS index from %s", FAISS_PATH)
            self.vectorstore = FAISS.load_local(
                FAISS_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            logger.info("No FAISS index found — ingesting documents …")
            self.ingest_documents()

        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10},
        )

        self.llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )

        logger.info("RAG pipeline ready.")

    # ------------------------------------------------------------------ #
    # Document Ingestion
    # ------------------------------------------------------------------ #

    def ingest_documents(self) -> int:
        logger.info("Loading documents from %s", DATA_PATH)

        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents = loader.load()
        if not documents:
            raise FileNotFoundError(
                f"No .pdf documents found in {DATA_PATH}. "
                "Add HR policy PDF files and retry."
            )

        logger.info("Loaded %d documents. Splitting …", len(documents))

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Created %d chunks.", len(chunks))

        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        os.makedirs(FAISS_PATH, exist_ok=True)
        self.vectorstore.save_local(FAISS_PATH)
        logger.info("FAISS index saved to %s", FAISS_PATH)
        return len(chunks)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def chat(
        self,
        query: str,
        chat_history: List[Tuple[str, str]],
    ) -> dict:
        # Step 1: retrieve relevant chunks
        docs = self.retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        # Step 2: build message list
        system_prompt = (
            "You are HRBot, a knowledgeable and friendly HR assistant for "
            "Apex Technologies. Answer questions accurately based only on the "
            "provided HR policy context. "
            "If the answer is not in the context, say: "
            "'I don't have specific information on that in our HR policies. "
            "Please contact HR at hr@apextechnologies.com or call Ext. 1001.' "
            "Keep answers concise, clear, and professional. "
            "Format policy details as easy-to-read bullet points when helpful.\n\n"
            f"Context:\n{context}"
        )

        messages = [SystemMessage(content=system_prompt)]
        for human, ai in chat_history:
            messages.append(HumanMessage(content=human))
            messages.append(AIMessage(content=ai))
        messages.append(HumanMessage(content=query))

        # Step 3: call LLM
        response = self.llm.invoke(messages)

        sources = list({
            Path(doc.metadata.get("source", "HR Policy Manual")).name
            for doc in docs
        })

        return {
            "answer":  response.content,
            "sources": sources,
        }

    def get_collection_count(self) -> int:
        if self.vectorstore:
            return self.vectorstore.index.ntotal
        return 0
