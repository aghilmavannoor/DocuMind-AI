import streamlit as st
import fitz
import numpy as np
import os
import re

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


# =========================================================
# SETUP
# =========================================================

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


# =========================================================
# CHUNKING (paragraph-aware, with junk filtering)
# =========================================================

_STOPWORDS = {
    "the", "of", "and", "a", "to", "in", "is", "that", "it", "for", "on",
    "as", "with", "was", "his", "her", "he", "she", "you", "your", "are",
    "be", "at", "by", "an", "or", "this", "not", "have", "has", "had",
    "but", "from", "they", "we", "who", "what", "so", "if", "there",
    "when", "were", "been", "their", "them", "than", "then", "will",
}


def is_low_content(text):
    """Detect TOC lines, page-number lists, book-title lists, and other
    non-prose junk that would otherwise pollute the embedding index.

    Uses word-token-level ratios instead of raw character/length counts,
    because a long chunk (e.g. a whole merged TOC page) can dilute a
    character-based digit ratio even though it's still structurally
    non-prose (headings + scattered page numbers, not sentences).
    """
    stripped = text.strip()
    if len(stripped) < 40:
        return True

    words = re.findall(r"\b\w+\b", stripped)
    if not words:
        return True

    # TOC / index pages are full of standalone page-number tokens
    # ("15", "72", "97"...). Real prose rarely has more than a trailing
    # page number or two.
    digit_tokens = [w for w in words if w.isdigit()]
    digit_token_ratio = len(digit_tokens) / len(words)
    if digit_token_ratio > 0.06:
        return True

    # Real prose is dense with common function words. Heading lists and
    # book-title lists ("By X. Title One / Title Two / ...") are mostly
    # capitalized nouns strung together with few stopwords.
    stopword_count = sum(1 for w in words if w.lower() in _STOPWORDS)
    stopword_ratio = stopword_count / len(words)
    if stopword_ratio < 0.15:
        return True

    # Title-list detection. A book/author list capitalizes almost every
    # content word ("Making Love Last: Creating and Maintaining..."),
    # whereas real prose only capitalizes proper nouns and the first word
    # of each sentence. So only count capitalization *after* the first
    # word of a sentence - that isolates genuine title-casing from normal
    # sentence-initial capitals, which would otherwise cause false
    # positives on prose full of short sentences.
    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    content_word_count = 0
    midsentence_cap_count = 0

    for sentence in sentences:
        sentence_words = re.findall(r"[A-Za-z][A-Za-z\-']*", sentence)
        for idx, w in enumerate(sentence_words):
            if w.lower() in _STOPWORDS:
                continue
            content_word_count += 1
            if idx == 0:
                continue  # sentence-initial capital is normal, not a signal
            if w[0].isupper():
                midsentence_cap_count += 1

    if content_word_count:
        cap_ratio = midsentence_cap_count / content_word_count
        if cap_ratio > 0.35:
            return True

    return False


def split_text(text, page_number, target_size=1000):
    """Split on paragraph breaks first, then pack paragraphs up to
    target_size so sentences/ideas aren't cut mid-thought."""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    buffer = ""

    for para in paragraphs:
        if len(buffer) + len(para) <= target_size:
            buffer = (buffer + " " + para).strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para

    if buffer:
        chunks.append(buffer)

    result = []
    for c in chunks:
        if not is_low_content(c):
            result.append({"text": c, "page": page_number})

    return result


# =========================================================
# APP
# =========================================================

st.title("DocuMind AI")
st.write("Intelligent PDF Research Assistant")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:

    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    # Only reprocess the PDF if it's a new upload. Without this, Streamlit
    # re-runs the whole script (including re-embedding every chunk) on
    # every single question, which is slow and pointless.
    if st.session_state.get("file_id") != file_id:

        st.success("PDF uploaded successfully!")
        st.write("File name:", uploaded_file.name)
        st.write("File size:", uploaded_file.size, "bytes")

        with st.spinner("Reading and embedding PDF... (only happens once per upload)"):

            pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

            all_chunks = []
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text()
                all_chunks.extend(split_text(page_text, page_number))

            chunk_texts = [c["text"] for c in all_chunks]
            embeddings = model.encode(chunk_texts)

            st.session_state["file_id"] = file_id
            st.session_state["chunks"] = all_chunks
            st.session_state["embeddings"] = np.array(embeddings)
            st.session_state["num_pages"] = len(pdf)

    else:
        st.success("Using cached PDF (already processed).")

    chunks = st.session_state["chunks"]
    embeddings = st.session_state["embeddings"]

    st.write("Number of pages:", st.session_state["num_pages"])
    st.subheader("Text Chunks")
    st.write("Number of chunks:", len(chunks))

    st.subheader("Embeddings")
    st.write("Number of embeddings:", len(embeddings))
    st.write("Embedding dimensions:", embeddings.shape[1])

    # =====================================================
    # QUESTION
    # =====================================================

    question = st.text_input("Ask a question about your PDF:")
    top_k = st.slider("Number of chunks to retrieve", min_value=3, max_value=15, value=8)
    retrieval_only = st.checkbox(
        "Retrieval-only debug mode (skip Gemini, no API calls used)",
        value=False,
    )

    if question:

        question_embedding = model.encode(question)

        # Vectorized cosine similarity instead of a per-chunk Python loop
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
        norms[norms == 0] = 1e-10  # avoid divide-by-zero
        similarities = (embeddings @ question_embedding) / norms

        top_indexes = np.argsort(similarities)[-top_k:][::-1]

        # Drop very weak matches instead of always padding to top_k
        SIM_THRESHOLD = 0.30
        top_indexes = [i for i in top_indexes if similarities[i] >= SIM_THRESHOLD]

        # Neighbor expansion: a matched chunk is often the *start* of the
        # relevant passage, cut off mid-thought by our fixed chunk size.
        # Pull in the next chunk too (if it exists and isn't already
        # included) so the model sees the continuation, not just the
        # opening fragment.
        expanded_indexes = []
        seen = set()
        for index in top_indexes:
            for i in (index, index + 1):
                if 0 <= i < len(chunks) and i not in seen:
                    expanded_indexes.append(i)
                    seen.add(i)

        st.subheader("Most Relevant Chunks")

        relevant_chunks = []

        if not expanded_indexes:
            st.write("No sufficiently relevant chunks found.")
        else:
            for index in expanded_indexes:
                is_expansion = index not in top_indexes
                label = " (continued)" if is_expansion else ""
                st.write(f"📖 Page: {chunks[index]['page']}{label}")
                if not is_expansion:
                    st.write(f"⭐ Similarity Score: {similarities[index]:.4f}")
                st.text(chunks[index]["text"])
                relevant_chunks.append(chunks[index]["text"])

        context = "\n\n".join(relevant_chunks)

        prompt = f"""
You are a helpful document assistant.

Answer the user's question using only the information provided in the
context below. The context is made of several excerpts retrieved from
different parts of a document, so it may be incomplete or only partially
address the question - that is normal and expected.

Do your best to construct a real answer from whatever relevant material
is present, even if it's fragmentary or only partially covers the topic.
Only say "I couldn't find the answer in the document" if the context is
genuinely unrelated to the question - not merely because it's incomplete.

Context:
{context}

Question:
{question}

Answer:
"""

        if retrieval_only:
            st.subheader("AI Answer")
            st.info(
                "Retrieval-only mode is on — skipping the Gemini call. "
                "Review the ranked chunks above to judge retrieval quality "
                "on their own, before generation is involved at all."
            )
        else:
            answer_cache_key = f"answer::{file_id}::{question}::{tuple(expanded_indexes)}"
            cached = st.session_state.get(answer_cache_key)

            if cached is not None:
                answer_text = cached
                st.caption("(Answer reused from cache - identical question already asked.)")
            else:
                try:
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt
                    )
                    answer_text = response.text
                    st.session_state[answer_cache_key] = answer_text
                except Exception as e:
                    error_str = str(e)
                    if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                        answer_text = (
                            "The Gemini API's free-tier daily quota has been used up "
                            "for this model. It resets roughly every 24 hours, or you "
                            "can enable billing on your Google AI Studio project to "
                            "raise the limit. (Raw error details are in the terminal/logs.)"
                        )
                        print(f"Gemini quota error: {error_str}")
                    else:
                        answer_text = f"Error contacting the AI model: {e}"

            st.subheader("AI Answer")
            st.write(answer_text)

            # Lightweight safety note for sensitive content, without diagnosing
            # or overriding the model's actual answer.
            sensitive_terms = ["suicide", "suicidal", "self-harm", "crisis"]
            if any(term in context.lower() for term in sensitive_terms):
                st.info(
                    "This document touches on sensitive mental health topics. "
                    "If you or someone you know is struggling, please reach out "
                    "to a mental health professional or a crisis line."
                )