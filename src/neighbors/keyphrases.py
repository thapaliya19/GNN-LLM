"""
Keyphrase extraction for node text descriptions.
Based on GraphiT (Khoshraftar et al., WWW 2025): uses transformer-based
semantic keyphrase extraction with MMR diversity filtering.
"""

from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm


def extract_keyphrases_for_corpus(
    texts: List[str],
    method: str = "keybert",
    top_k: int = 5,
    sbert_model=None,
) -> Dict[int, List[str]]:
    """Extract keyphrases for all node texts in the corpus.

    Args:
        texts: List of node text descriptions (indexed by node_id).
        method: "keybert" (semantic, preferred) or "tfidf" (fallback).
        top_k: Number of keyphrases per node.
        sbert_model: Pre-loaded SentenceTransformer model (reused for KeyBERT).

    Returns:
        Dict mapping node index to list of keyphrase strings.
    """
    if method == "keybert":
        return _extract_keybert(texts, top_k, sbert_model)
    elif method == "tfidf":
        return _extract_tfidf(texts, top_k)
    else:
        raise ValueError(f"Unknown keyphrase method: {method}")


def _extract_keybert(
    texts: List[str], top_k: int, sbert_model=None
) -> Dict[int, List[str]]:
    """KeyBERT: transformer-based semantic keyphrase extraction with MMR diversity."""
    try:
        from keybert import KeyBERT
    except ImportError:
        print("[KEYPHRASE] keybert not installed, falling back to tfidf")
        return _extract_tfidf(texts, top_k)

    # Reuse the already-loaded SBERT model if provided
    kw_model = KeyBERT(model=sbert_model) if sbert_model is not None else KeyBERT()

    keyphrases_by_node: Dict[int, List[str]] = {}
    for i, text in enumerate(tqdm(texts, desc="[KEYPHRASE] Extracting (KeyBERT)")):
        if not text or len(text.split()) < 3:
            keyphrases_by_node[i] = [text.strip()] if text and text.strip() else ["(no text)"]
            continue
        try:
            kws = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=top_k,
                use_mmr=True,
                diversity=0.5,
            )
            keyphrases_by_node[i] = [kw for kw, _ in kws] if kws else [text[:50]]
        except Exception:
            # Fallback for problematic texts
            keyphrases_by_node[i] = [text[:50]]

    return keyphrases_by_node


def _extract_tfidf(texts: List[str], top_k: int) -> Dict[int, List[str]]:
    """TF-IDF fallback: corpus-level fitting, per-node top-k extraction."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Replace empty texts with placeholder to avoid vectorizer issues
    clean_texts = [t if t and t.strip() else "(empty)" for t in texts]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=10000,
    )
    tfidf_matrix = vectorizer.fit_transform(clean_texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    keyphrases_by_node: Dict[int, List[str]] = {}
    for i in tqdm(range(len(texts)), desc="[KEYPHRASE] Extracting (TF-IDF)"):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_k]
        kws = [feature_names[idx] for idx in top_indices if row[idx] > 0]
        keyphrases_by_node[i] = kws if kws else ["(no keyphrases)"]

    return keyphrases_by_node


def format_keyphrases(keyphrases: List[str]) -> str:
    """Format keyphrases as a bracketed comma-separated string."""
    return "[" + ", ".join(keyphrases) + "]"
