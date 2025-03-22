import re
import math
import nltk
from typing import List, Callable, Union

class Chunker:
    """
    Chunker class for semantic chunking of documents for RAG.
    Splits a text into sentences, then groups sentences into semantically coherent chunks.
    """
    def __init__(self, embedding_func: Callable[[str], List[float]], similarity_metric: str = 'cosine', chunk_size: int = 5, overlap: Union[int, float] = 0):
        """
        Initialize the Chunker with an embedding function and optional settings.

        :param embedding_func: function that takes a sentence (str) and returns its embedding (List[float]).
        :param similarity_metric: 'cosine' (default) or 'jaccard' to measure similarity between sentences.
        :param chunk_size: desired number of sentences per chunk (maximum, in semantic mode).
        :param overlap: overlap between consecutive chunks (int for number of sentences, float for fraction of chunk_size).
        """
        self.embedding_func = embedding_func
        self.similarity_metric = similarity_metric.lower()
        self.chunk_size = chunk_size
        # Convert overlap to an integer number of sentences (if float, treat as percentage of chunk_size)
        if isinstance(overlap, float):
            if overlap < 1.0:
                # e.g., 0.2 means 20% of chunk_size
                self.overlap = int(round(overlap * chunk_size))
            else:
                # If float >= 1, treat it as number of sentences (though float here is unusual)
                self.overlap = int(round(overlap))
        else:
            self.overlap = overlap
        # Ensure overlap is less than chunk_size to avoid infinite loop
        if self.overlap >= self.chunk_size:
            self.overlap = self.chunk_size - 1 if self.chunk_size > 0 else 0
        # Internal default similarity threshold for deciding chunk breaks (for semantic grouping)
        # These can be adjusted for different datasets or metrics
        self._cosine_threshold = 0.8   # high threshold: require strong similarity to stay in chunk
        self._jaccard_threshold = 0.3  # Jaccard tends to be lower since it's harder to have large overlap

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split the input text into a list of sentences.
        Uses NLTK Punkt tokenizer if available for language-independent splitting.
        Falls back to a simple rule-based split if necessary.
        """
        try:
            # Attempt to use NLTK's sentence tokenizer (Punkt)
            # Ensure the Punkt tokenizer models are downloaded (for the first run)
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            sent_tokens = nltk.tokenize.sent_tokenize(text)
            return [s.strip() for s in sent_tokens if s.strip()]
        except ImportError:
            # NLTK not available, use a simple fallback (not as accurate for all languages)
            # Regex to split on punctuation that likely indicates end of sentence.
            # This is a naive approach and may not handle all cases (e.g., "Mr. Smith").
            sentences = re.split(r'(?<=[\.!\?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        # Dot product
        dot = 0.0
        for a, b in zip(vec1, vec2):
            dot += a * b
        # Norms
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            # If either vector has zero magnitude, define similarity as 0 to avoid division by zero
            return 0.0
        return dot / (norm1 * norm2)

    def _jaccard_similarity(self, sent1: str, sent2: str) -> float:
        """Compute Jaccard similarity between two sentences (as sets of words)."""
        # Tokenize sentences into words (simple split by non-alphanumeric characters)
        words1 = set(filter(None, re.split(r'\W+', sent1.lower())))
        words2 = set(filter(None, re.split(r'\W+', sent2.lower())))
        if not words1 or not words2:
            return 0.0
        # Intersection and union
        inter = words1.intersection(words2)
        union = words1.union(words2)
        return len(inter) / len(union)

    def _are_sentences_similar(self, prev_sent: str, next_sent: str,
                                prev_emb: List[float] = None, next_emb: List[float] = None) -> bool:
        """
        Decide if next_sent is semantically similar enough to prev_sent to be in the same chunk.
        Uses the chosen similarity metric and threshold.
        """
        if self.similarity_metric == 'cosine':
            # Ensure we have embeddings (should be precomputed and passed in to avoid recomputation)
            if prev_emb is None:
                prev_emb = self.embedding_func(prev_sent)
            if next_emb is None:
                next_emb = self.embedding_func(next_sent)
            sim = self._cosine_similarity(prev_emb, next_emb)
            return sim >= self._cosine_threshold
        elif self.similarity_metric == 'jaccard':
            # Compute Jaccard on the fly using sentences (embeddings not required for jaccard)
            sim = self._jaccard_similarity(prev_sent, next_sent)
            return sim >= self._jaccard_threshold
        else:
            # If an unknown metric is set, default to False (or we could default to cosine)
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")

    def semantic_chunking(self, corpus: str, chunk_size: int = None, overlap: Union[int, float] = None) -> List[str]:
        """
        Perform semantic chunking on the given corpus text.

        :param corpus: The full text to be chunked.
        :param chunk_size: (Optional) override for number of sentences per chunk.
        :param overlap: (Optional) override for overlap (int for count, float for percentage).
        :return: List of chunks (each chunk is a string of one or more sentences).
        """
        # Use the instance's default chunk_size and overlap if not provided
        chunk_size = chunk_size or self.chunk_size
        overlap_val = overlap if overlap is not None else self.overlap
        # Recompute overlap_count if an override is given
        if overlap is not None:
            if isinstance(overlap_val, float):
                if overlap_val < 1.0:
                    overlap_count = int(round(overlap_val * chunk_size))
                else:
                    overlap_count = int(round(overlap_val))
            else:
                overlap_count = overlap_val
            if overlap_count >= chunk_size:
                overlap_count = chunk_size - 1 if chunk_size > 0 else 0
        else:
            overlap_count = self.overlap  # use already calculated self.overlap from __init__

        # 1. Split the text into sentences
        sentences = self._split_sentences(corpus)
        if not sentences:
            return []

        # 2. Precompute embeddings for all sentences if using cosine (to avoid repeated calls)
        embeddings = []
        if self.similarity_metric == 'cosine':
            embeddings = [self.embedding_func(sent) for sent in sentences]
        else:
            # For Jaccard, embeddings list can remain empty or None (not used)
            embeddings = [None] * len(sentences)

        chunks: List[List[str]] = []  # will hold list of sentences for each chunk
        current_chunk: List[str] = []
        current_chunk_embeds: List[List[float]] = []  # parallel list of embeddings for current chunk

        # 3. Iterate through sentences and group into chunks
        for idx, sentence in enumerate(sentences):
            if not current_chunk:
                # Start a new chunk with the current sentence
                current_chunk.append(sentence)
                if self.similarity_metric == 'cosine':
                    current_chunk_embeds.append(embeddings[idx])
                else:
                    current_chunk_embeds.append(None)  # placeholder for consistency
                continue

            # Check similarity with previous sentence in the current chunk
            prev_sent = current_chunk[-1]
            prev_emb = current_chunk_embeds[-1] if self.similarity_metric == 'cosine' else None
            next_sent = sentence
            next_emb = embeddings[idx] if self.similarity_metric == 'cosine' else None

            # Decide if we should break the chunk here
            # Break if current chunk is already at max size, or if the sentences are not similar enough
            if len(current_chunk) >= chunk_size or not self._are_sentences_similar(prev_sent, next_sent, prev_emb, next_emb):
                # Finalize the current chunk
                chunks.append(current_chunk)
                # Start a new chunk, considering overlap
                if overlap_count > 0:
                    # Carry over the last `overlap_count` sentences from the old chunk into the new chunk
                    overlap_sents = current_chunk[-overlap_count:]
                    current_chunk = list(overlap_sents)  # start new chunk with the overlap part
                    if self.similarity_metric == 'cosine':
                        overlap_embeds = current_chunk_embeds[-overlap_count:]
                    else:
                        overlap_embeds = [None] * len(overlap_sents)
                    current_chunk_embeds = list(overlap_embeds)
                else:
                    # No overlap: start a completely fresh chunk
                    current_chunk = []
                    current_chunk_embeds = []
                # Now add the new sentence as the beginning of the next chunk
                current_chunk.append(sentence)
                if self.similarity_metric == 'cosine':
                    current_chunk_embeds.append(embeddings[idx])
                else:
                    current_chunk_embeds.append(None)
            else:
                # If similar and under size limit, add sentence to current chunk
                current_chunk.append(sentence)
                if self.similarity_metric == 'cosine':
                    current_chunk_embeds.append(embeddings[idx])
                else:
                    current_chunk_embeds.append(None)

        # After loop, add the final chunk if it has content
        if current_chunk:
            chunks.append(current_chunk)

        # 4. Join sentences in each chunk back into a single string (preserve original punctuation spacing)
        chunk_texts: List[str] = []
        for chunk in chunks:
            # Reconstruct chunk text. Here we simply join with a space.
            # If needed, this could be improved to ensure proper spacing/punctuation (based on original text).
            text = " ".join(chunk)
            chunk_texts.append(text)

        return chunk_texts
