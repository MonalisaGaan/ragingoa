import os
import time
from pathlib import Path

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

from .rag import RAGEngine
from .guardrails import (
    validate_query,
    check_relevance,
    build_context,
    grounding_check
)


# =========================================================
# PATHS / ENVIRONMENT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="RAGVoice",
    description="Voice-enabled grounded RAG using MSMARCO-XI",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# =========================================================
# RAG ENGINE
# =========================================================

print("Initializing RAG engine...")

rag = RAGEngine()


# =========================================================
# DEMO KNOWLEDGE
# =========================================================

DEMO_DOCUMENTS = [

    {
        "id": "rag-001",
        "text": """
        Retrieval augmented generation combines
        information retrieval with language generation.
        A retriever searches a knowledge base for
        relevant information and the language model
        uses that retrieved context to generate an answer.
        """
    },

    {
        "id": "rag-002",
        "text": """
        FAISS is a library for efficient similarity
        search over dense vectors. Text can be converted
        into embeddings and stored in an index.
        A query can then be embedded and compared against
        those stored vectors.
        """
    },

    {
        "id": "rag-003",
        "text": """
        Chunking divides documents into smaller pieces
        before embedding. Fixed-size chunking is simple,
        overlapping chunking preserves context between
        boundaries, and sentence-aware chunking attempts
        to preserve meaningful semantic units.
        """
    },

    {
        "id": "rag-004",
        "text": """
        A grounded RAG system should avoid hallucination.
        If retrieved evidence is insufficient, the system
        should tell the user that it does not have enough
        information rather than inventing an answer.
        """
    },

    {
        "id": "rag-005",
        "text": """
        Latency should be evaluated using multiple
        measurements rather than a single best case.
        P50 represents the median latency, P70 represents
        a higher percentile, and P100 represents the
        slowest observed request in the benchmark.
        """
    }

]


# =========================================================
# INDEX DOCUMENTS
# =========================================================

for document in DEMO_DOCUMENTS:

    rag.add_document(
        document["text"],
        {
            "id": document["id"],
            "source": "MSMARCO-XI-demo",
            "language": "en"
        }
    )


rag.build_index()

print(
    "Indexed",
    len(rag.documents),
    "chunks"
)


# =========================================================
# REQUEST MODEL
# =========================================================

class QueryRequest(BaseModel):

    query: str


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "online",
        "service": "RAGVoice",
        "indexed_chunks": len(rag.documents)
    }


# =========================================================
# RAG QUERY
# =========================================================

@app.post("/query")
def query(request: QueryRequest):

    start = time.perf_counter()


    # -----------------------------------------------------
    # VALIDATE QUERY
    # -----------------------------------------------------

    valid, reason = validate_query(
        request.query
    )


    if not valid:

        return {
            "answer": reason,

            "grounded": False,

            "status": "blocked",

            "latency_ms": round(
                (
                    time.perf_counter()
                    - start
                ) * 1000,
                2
            ),

            "sources": []
        }


    # -----------------------------------------------------
    # RETRIEVE
    # -----------------------------------------------------

    results, retrieval_latency = rag.retrieve(
        request.query,
        top_k=5
    )


    # -----------------------------------------------------
    # CHECK RELEVANCE
    # -----------------------------------------------------

    if not check_relevance(results):

        return {
            "answer": (
                "I couldn't find enough "
                "relevant evidence in the "
                "knowledge base to answer "
                "that reliably."
            ),

            "grounded": False,

            "status": "insufficient_context",

            "retrieval_latency_ms": round(
                retrieval_latency,
                2
            ),

            "latency_ms": round(
                (
                    time.perf_counter()
                    - start
                ) * 1000,
                2
            ),

            "sources": []
        }


    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    context = build_context(
        results
    )


    # -----------------------------------------------------
    # GROUNDED MVP ANSWER
    # -----------------------------------------------------

    answer = (
        "Based on the retrieved evidence: "
        + results[0].get(
            "text",
            "No evidence text available."
        )
    )


    # -----------------------------------------------------
    # GROUNDING CHECK
    # -----------------------------------------------------

    grounded = grounding_check(
        answer,
        context
    )


    # -----------------------------------------------------
    # TOTAL LATENCY
    # -----------------------------------------------------

    total_latency = (
        time.perf_counter()
        - start
    ) * 1000


    # -----------------------------------------------------
    # RETURN RESPONSE
    #
    # IMPORTANT:
    # result dictionaries may not contain "chunk_id".
    # Therefore .get() is used instead of result["chunk_id"].
    # -----------------------------------------------------

    return {

        "answer": answer,

        "grounded": grounded,

        "status": (
            "grounded"
            if grounded
            else "review"
        ),

        "retrieval_latency_ms": round(
            retrieval_latency,
            2
        ),

        "latency_ms": round(
            total_latency,
            2
        ),

        "sources": [

            {

                "chunk_id": result.get(
                    "chunk_id",
                    result.get(
                        "id",
                        "unknown"
                    )
                ),

                "strategy": result.get(
                    "strategy",
                    "retrieval"
                ),

                "score": round(
                    float(
                        result.get(
                            "score",
                            0
                        )
                    ),
                    4
                ),

                "text": result.get(
                    "text",
                    ""
                )

            }

            for result in results

        ]

    }


# =========================================================
# SARVAM SPEECH TO TEXT
# =========================================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...)
):

    api_key = os.getenv(
        "SARVAM_API_KEY"
    )


    if not api_key:

        raise HTTPException(
            status_code=500,
            detail=(
                "SARVAM_API_KEY is missing "
                "from .env"
            )
        )


    # -----------------------------------------------------
    # READ AUDIO
    # -----------------------------------------------------

    audio_bytes = await file.read()


    if not audio_bytes:

        raise HTTPException(
            status_code=400,
            detail="Empty audio file."
        )


    # -----------------------------------------------------
    # TEMPORARY AUDIO FILE
    # -----------------------------------------------------

    temp_file = (
        BASE_DIR /
        "temporary_voice.webm"
    )


    with open(
        temp_file,
        "wb"
    ) as audio_file:

        audio_file.write(
            audio_bytes
        )


    try:

        # -------------------------------------------------
        # SARVAM SDK
        # -------------------------------------------------

        from sarvamai import SarvamAI


        client = SarvamAI(
            api_subscription_key=api_key
        )


        response = (
            client
            .speech_to_text
            .transcribe(
                file=open(
                    temp_file,
                    "rb"
                ),
                model="saaras:v3"
            )
        )


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {

            "transcript":
                response.transcript,

            "language":
                getattr(
                    response,
                    "language_code",
                    None
                )

        }


    except Exception as error:

        print(
            "Sarvam transcription error:",
            repr(error)
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "Speech transcription failed: "
                + str(error)
            )
        )


    finally:

        # -------------------------------------------------
        # DELETE TEMP FILE
        # -------------------------------------------------

        if temp_file.exists():

            temp_file.unlink()


# =========================================================
# FRONTEND
# =========================================================

FRONTEND_DIR = BASE_DIR / "frontend"


if FRONTEND_DIR.is_dir():

    app.mount(
        "/",
        StaticFiles(
            directory=str(
                FRONTEND_DIR
            ),
            html=True
        ),
        name="frontend"
    )

else:

    print(
        "WARNING: frontend directory not found:",
        FRONTEND_DIR
    )