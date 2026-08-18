import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .rag import RAGEngine
from .guardrails import (
    validate_query,
    check_relevance,
    build_context,
    grounding_check,
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
    description="Voice-enabled grounded Retrieval-Augmented Generation",
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# RAG ENGINE
# =========================================================

print("Initializing RAG engine...")

rag = RAGEngine()


# =========================================================
# LOCAL DEMO KNOWLEDGE BASE
# =========================================================

DEMO_DOCUMENTS = [

    {
        "id": "ai-001",
        "topic": "Artificial Intelligence",
        "text": """
Artificial intelligence, or AI, is a field of computer science
focused on creating systems that can perform tasks that normally
require human intelligence. These tasks include understanding
language, recognizing patterns, learning from data, reasoning,
and making decisions.
""",
    },

    {
        "id": "ml-001",
        "topic": "Machine Learning",
        "text": """
Machine learning is a branch of artificial intelligence in which
computers learn patterns from data instead of being explicitly
programmed for every individual task. Common types include
supervised learning, unsupervised learning, and reinforcement
learning. Machine learning is used for prediction, classification,
recommendation systems, and many other applications.
""",
    },

    {
        "id": "dl-001",
        "topic": "Deep Learning",
        "text": """
Deep learning is a type of machine learning that uses neural
networks with multiple layers. Deep learning models can learn
complex patterns from large amounts of data and are widely used
for image recognition, speech recognition, natural language
processing, and computer vision.
""",
    },

    {
        "id": "rag-001",
        "topic": "Retrieval Augmented Generation",
        "text": """
Retrieval Augmented Generation, commonly called RAG, combines
information retrieval with language generation. A RAG system
first searches a knowledge base for relevant information and
then uses the retrieved information as context to produce a
grounded answer. This can reduce unsupported answers and help
a system work with external knowledge.
""",
    },

    {
        "id": "rag-002",
        "topic": "RAG Retrieval",
        "text": """
A RAG retrieval component searches a collection of documents
to find passages relevant to a user's question. Keyword-based
methods such as BM25 can identify matching terms, while TF-IDF
can measure lexical similarity between a query and documents.
Hybrid retrieval can combine multiple relevance signals.
""",
    },

    {
        "id": "chunk-001",
        "topic": "Document Chunking",
        "text": """
Document chunking divides large documents into smaller pieces
before they are indexed or embedded. Fixed-size chunking uses
a defined number of characters or tokens. Overlapping chunks
preserve information across boundaries. Sentence-aware or
semantic chunking attempts to keep related ideas together.
""",
    },

    {
        "id": "faiss-001",
        "topic": "FAISS",
        "text": """
FAISS is a library designed for efficient similarity search.
It is commonly used with vector embeddings. Documents can be
converted into numerical vectors and stored in an index, after
which a query vector can be compared against those vectors to
retrieve similar documents.
""",
    },

    {
        "id": "python-001",
        "topic": "Python",
        "text": """
Python is a high-level programming language known for readable
syntax and a large ecosystem of libraries. It is widely used for
web development, automation, data analysis, artificial
intelligence, machine learning, and scientific computing.
""",
    },

    {
        "id": "fastapi-001",
        "topic": "FastAPI",
        "text": """
FastAPI is a Python web framework designed for building APIs.
It provides automatic API documentation, request validation
through Python type hints and Pydantic, and support for
asynchronous programming. FastAPI is commonly used for backend
services and machine learning APIs.
""",
    },

    {
        "id": "django-001",
        "topic": "Django",
        "text": """
Django is a Python web framework that follows a batteries-included
approach. It provides features such as URL routing, database
models, authentication, templates, forms, and an administrative
interface. Django is commonly used for building complete web
applications.
""",
    },

    {
        "id": "cloud-001",
        "topic": "Cloud Computing",
        "text": """
Cloud computing provides computing resources such as servers,
storage, databases, networking, and software over the internet.
Instead of maintaining all infrastructure locally, organizations
can use cloud services and scale resources according to demand.
""",
    },

    {
        "id": "cyber-001",
        "topic": "Cybersecurity",
        "text": """
Cybersecurity is the practice of protecting computers, networks,
applications, and data from unauthorized access, attacks, and
disruption. Important areas include authentication, access
control, encryption, network security, vulnerability management,
monitoring, and incident response.
""",
    },

    {
        "id": "phishing-001",
        "topic": "Phishing",
        "text": """
Phishing is a type of social engineering attack in which an
attacker attempts to trick a person into revealing sensitive
information or performing an unsafe action. Phishing messages
often imitate trusted organizations and may use misleading
links, urgent language, or fake login pages.
""",
    },

    {
        "id": "ransomware-001",
        "topic": "Ransomware",
        "text": """
Ransomware is malicious software that can prevent users or
organizations from accessing their data or systems. Organizations
reduce ransomware risk through backups, security updates,
least-privilege access, network monitoring, endpoint protection,
and security awareness training.
""",
    },

    {
        "id": "internet-001",
        "topic": "Internet",
        "text": """
The internet is a global network of interconnected computer
networks. Devices communicate using standardized protocols such
as TCP/IP. Services including websites, email, video streaming,
and online applications operate over this network.
""",
    },

    {
        "id": "http-001",
        "topic": "HTTP",
        "text": """
HTTP, or Hypertext Transfer Protocol, is an application-layer
protocol used for communication between clients and servers.
A client sends a request and a server returns a response.
Common HTTP methods include GET, POST, PUT, PATCH, and DELETE.
""",
    },

    {
        "id": "database-001",
        "topic": "Databases",
        "text": """
A database is an organized collection of data that can be
stored, managed, and retrieved efficiently. Relational databases
organize information into tables and commonly use SQL.
Examples include PostgreSQL, MySQL, and SQLite.
""",
    },

    {
        "id": "sql-001",
        "topic": "SQL",
        "text": """
SQL stands for Structured Query Language. It is used to interact
with relational databases. SQL can be used to retrieve, insert,
update, and delete data. Common SQL operations include SELECT,
INSERT, UPDATE, and DELETE.
""",
    },

    {
        "id": "photosynthesis-001",
        "topic": "Photosynthesis",
        "text": """
Photosynthesis is a biological process used by plants, algae,
and some microorganisms to convert light energy into chemical
energy. Plants use carbon dioxide and water and, with light
energy, produce glucose while releasing oxygen.
""",
    },

    {
        "id": "earthquake-001",
        "topic": "Earthquakes",
        "text": """
Earthquakes occur when energy is suddenly released within the
Earth's crust, often because rocks move along geological faults.
The released energy travels through the Earth as seismic waves.
The strength of an earthquake can be measured using different
scales and instruments.
""",
    },

    {
        "id": "solar-001",
        "topic": "Solar Energy",
        "text": """
Solar energy is energy obtained from sunlight. Solar photovoltaic
systems convert sunlight directly into electricity, while solar
thermal systems use sunlight to produce heat. Solar energy is
renewable because sunlight is naturally replenished.
""",
    },

    {
        "id": "renewable-001",
        "topic": "Renewable Energy",
        "text": """
Renewable energy comes from sources that are naturally replenished,
including sunlight, wind, flowing water, geothermal heat, and
biomass. Renewable energy can reduce dependence on finite fossil
fuel resources and can help reduce greenhouse gas emissions.
""",
    },

    {
        "id": "space-001",
        "topic": "Space",
        "text": """
Space is the region beyond Earth's atmosphere. It contains
planets, stars, galaxies, asteroids, comets, and other objects.
Our solar system contains the Sun and the objects that orbit it,
including Earth and other planets.
""",
    },

    {
        "id": "gravity-001",
        "topic": "Gravity",
        "text": """
Gravity is a fundamental force associated with mass. It causes
objects to attract one another. Earth's gravity gives objects
weight and keeps the Moon in orbit around Earth while Earth's
motion around the Sun is influenced by the Sun's gravity.
""",
    },

    {
        "id": "water-001",
        "topic": "Water Cycle",
        "text": """
The water cycle describes the continuous movement of water through
the environment. Important processes include evaporation,
condensation, precipitation, infiltration, and runoff. Solar
energy drives much of the cycle.
""",
    },

]


# =========================================================
# INDEX LOCAL KNOWLEDGE
# =========================================================

print(
    "Loading local RAG knowledge base..."
)

for document in DEMO_DOCUMENTS:

    rag.add_document(
        document["text"],
        {
            "id": document["id"],
            "topic": document["topic"],
            "source": "RAGVoice Demo Knowledge Base",
        },
    )


rag.build_index()

print(
    "RAG index ready."
)

print(
    "Indexed",
    len(rag.documents),
    "knowledge chunks."
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
        "retrieval": "BM25 + TF-IDF",
        "indexed_documents": len(rag.documents),
    }


# =========================================================
# QUERY ENDPOINT
# =========================================================

@app.post("/query")
def query(request: QueryRequest):

    start = time.perf_counter()

    user_query = request.query.strip()


    # =====================================================
    # VALIDATE
    # =====================================================

    valid, reason = validate_query(
        user_query
    )

    if not valid:

        return {
            "answer": reason,
            "grounded": False,
            "status": "blocked",
            "retrieval_latency_ms": 0,
            "latency_ms": round(
                (
                    time.perf_counter()
                    - start
                ) * 1000,
                2,
            ),
            "sources": [],
        }


    # =====================================================
    # RETRIEVE
    # =====================================================

    results, retrieval_latency = rag.retrieve(
        user_query,
        top_k=5,
    )


    # =====================================================
    # CHECK RELEVANCE
    # =====================================================

    if not check_relevance(
        results,
        threshold=0.25,
    ):

        return {
            "answer": (
                "I couldn't find enough relevant "
                "evidence in my knowledge base to "
                "answer that reliably."
            ),
            "grounded": False,
            "status": "insufficient_context",
            "retrieval_latency_ms": round(
                retrieval_latency,
                2,
            ),
            "latency_ms": round(
                (
                    time.perf_counter()
                    - start
                ) * 1000,
                2,
            ),
            "sources": [],
        }


    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    context = build_context(
        results,
        max_chars=5000,
    )


    # =====================================================
    # GENERATE GROUNDED ANSWER
    # =====================================================

    best_result = results[0]

    answer = best_result.get(
        "text",
        "",
    ).strip()


    if not answer:

        answer = (
            "I found relevant evidence, "
            "but no answer text was available."
        )


    # =====================================================
    # GROUNDING CHECK
    # =====================================================

    grounded = grounding_check(
        answer,
        context,
    )


    # =====================================================
    # TOTAL LATENCY
    # =====================================================

    total_latency = (
        time.perf_counter()
        - start
    ) * 1000


    # =====================================================
    # RETURN RESPONSE
    # =====================================================

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
            2,
        ),

        "latency_ms": round(
            total_latency,
            2,
        ),

        "sources": [

            {
                "chunk_id": result.get(
                    "metadata",
                    {},
                ).get(
                    "id",
                    "unknown",
                ),

                "strategy": (
                    "hybrid_bm25_tfidf"
                ),

                "score": round(
                    float(
                        result.get(
                            "score",
                            0,
                        )
                    ),
                    4,
                ),

                "text": result.get(
                    "text",
                    "",
                ),
            }

            for result in results

        ],
    }


# =========================================================
# SARVAM SPEECH-TO-TEXT
# =========================================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
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
            ),
        )


    # =====================================================
    # READ AUDIO
    # =====================================================

    audio_bytes = await file.read()

    if not audio_bytes:

        raise HTTPException(
            status_code=400,
            detail="Empty audio file.",
        )


    # =====================================================
    # TEMPORARY AUDIO FILE
    # =====================================================

    temp_file = (
        BASE_DIR /
        "temporary_voice.webm"
    )

    with open(
        temp_file,
        "wb",
    ) as audio_file:

        audio_file.write(
            audio_bytes
        )


    try:

        # =================================================
        # SARVAM
        # =================================================

        from sarvamai import SarvamAI

        client = SarvamAI(
            api_subscription_key=api_key
        )

        with open(
            temp_file,
            "rb",
        ) as audio_file:

            response = (
                client
                .speech_to_text
                .transcribe(
                    file=audio_file,
                    model="saaras:v3",
                )
            )


        # =================================================
        # RESPONSE
        # =================================================

        return {

            "transcript":
                response.transcript,

            "language":
                getattr(
                    response,
                    "language_code",
                    None,
                ),

        }


    except Exception as error:

        print(
            "Sarvam transcription error:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Speech transcription failed: "
                + str(error)
            ),
        )


    finally:

        # =================================================
        # CLEAN TEMP FILE
        # =================================================

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
            html=True,
        ),
        name="frontend",
    )

else:

    print(
        "WARNING: frontend directory not found:",
        FRONTEND_DIR,
    )