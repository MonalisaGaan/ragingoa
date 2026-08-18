const recordButton = document.getElementById("recordButton");
const recordText = document.getElementById("recordText");
const recordStatus = document.getElementById("recordStatus");

const queryInput = document.getElementById("query");
const askButton = document.getElementById("askButton");

const results = document.getElementById("results");
const transcript = document.getElementById("transcript");
const answer = document.getElementById("answer");
const grounded = document.getElementById("grounded");

const retrievalLatency = document.getElementById("retrievalLatency");
const totalLatency = document.getElementById("totalLatency");
const sourceCount = document.getElementById("sourceCount");
const sources = document.getElementById("sources");


// =====================================================
// BROWSER SPEECH RECOGNITION
// =====================================================

const SpeechRecognition =
    window.SpeechRecognition ||
    window.webkitSpeechRecognition;

let recognition = null;
let listening = false;

if (SpeechRecognition) {

    recognition = new SpeechRecognition();

    recognition.lang = "en-IN";
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = function () {

        listening = true;

        recordButton.classList.add("recording");

        recordText.textContent = "Stop speaking";

        recordStatus.textContent =
            "Listening... Speak your question.";

    };


    recognition.onresult = function (event) {

        const text =
            event.results[0][0].transcript;

        queryInput.value = text;

        recordStatus.textContent =
            "Voice captured. Click Ask.";

    };


    recognition.onerror = function (event) {

        console.error("Speech recognition error:", event.error);

        recordStatus.textContent =
            "Microphone error: " + event.error;

    };


    recognition.onend = function () {

        listening = false;

        recordButton.classList.remove("recording");

        recordText.textContent = "Start speaking";

    };

} else {

    recordStatus.textContent =
        "Speech recognition is not supported in this browser.";

    recordButton.disabled = true;
}


// =====================================================
// MICROPHONE BUTTON
// =====================================================

recordButton.addEventListener("click", function () {

    if (!recognition) {
        return;
    }

    if (listening) {

        recognition.stop();

        return;
    }

    queryInput.focus();

    try {

        recognition.start();

    } catch (error) {

        console.error(error);

    }

});


// =====================================================
// ASK BUTTON
// =====================================================

askButton.addEventListener("click", askQuestion);


// =====================================================
// ENTER KEY
// =====================================================

queryInput.addEventListener("keydown", function (event) {

    if (event.key === "Enter") {

        event.preventDefault();

        askQuestion();

    }

});


// =====================================================
// ASK RAG
// =====================================================

async function askQuestion() {

    const question = queryInput.value.trim();

    if (!question) {

        recordStatus.textContent =
            "Please type or speak a question.";

        queryInput.focus();

        return;
    }

    recordStatus.textContent =
        "Retrieving grounded evidence...";

    askButton.disabled = true;

    results.classList.remove("hidden");

    transcript.textContent = question;

    answer.textContent =
        "Searching the knowledge base...";

    grounded.textContent = "";

    sources.innerHTML = "";

    try {

        const response = await fetch("/query", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: question
            })

        });

        const data = await response.json();

        if (!response.ok) {

            throw new Error(
                data.detail || "Backend error"
            );

        }

        answer.textContent =
            data.answer || "No answer returned.";

        grounded.textContent =
            data.grounded
                ? "✓ GROUNDED"
                : "⚠ REVIEW";

        retrievalLatency.textContent =
            (data.retrieval_latency_ms ?? 0) + " ms";

        totalLatency.textContent =
            (data.latency_ms ?? 0) + " ms";

        sourceCount.textContent =
            (data.sources || []).length;


        // =============================================
        // SOURCES
        // =============================================

        if (data.sources && data.sources.length > 0) {

            data.sources.forEach(function (source) {

                const div =
                    document.createElement("div");

                div.className = "source";

                div.innerHTML = `
                    <strong>
                        ${source.chunk_id || "Source"}
                    </strong>

                    <small>
                        Score: ${source.score ?? "N/A"}
                    </small>

                    <p>
                        ${source.text || ""}
                    </p>
                `;

                sources.appendChild(div);

            });

        } else {

            sources.textContent =
                "No sources returned.";

        }

        recordStatus.textContent =
            "Answer generated successfully.";

    } catch (error) {

        console.error("RAG error:", error);

        answer.textContent =
            "Could not connect to the RAG backend.";

        recordStatus.textContent =
            "Backend error. Check the Terminal.";

    } finally {

        askButton.disabled = false;

    }

}