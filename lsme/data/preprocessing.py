"""
Text preprocessing utilities for LSME.

Adapted from MDLM (https://github.com/kuleshov-group/mdlm) dataloader.py
detokenizer functions and preprocessing pipeline.
"""

import re


# --- Detokenizers adapted from MDLM ---

def wt_detokenizer(text):
    """Detokenizer for WikiText-style tokenization."""
    # Contractions
    text = text.replace("' ", "'").replace(" \n", "\n")
    text = text.replace(" n't", "n't").replace(" 'm", "'m")
    text = text.replace(" 's", "'s").replace(" 've", "'ve")
    text = text.replace(" 're", "'re")
    # Punctuation
    text = re.sub(r'\s([.,:;?!)\]}])', r'\1', text)
    text = re.sub(r'([\[({])\s', r'\1', text)
    text = text.replace(" - ", "-")
    return text.strip()


def ptb_detokenizer(text):
    """Detokenizer for Penn Treebank-style tokenization."""
    text = text.replace(" 's", "'s").replace("s ' ", "s' ")
    text = text.replace(" n't", "n't").replace(" 'm", "'m")
    text = text.replace(" 've", "'ve").replace(" 're", "'re")
    text = text.replace(" 'd", "'d").replace(" 'll", "'ll")
    text = re.sub(r'\s([.,:;?!)\]}])', r'\1', text)
    text = re.sub(r'([\[({])\s', r'\1', text)
    text = text.replace("` ", '"').replace(" '", '"')
    text = text.replace(" - ", "-")
    return text.strip()


_DETOKENIZERS = {
    "wikitext": wt_detokenizer,
    "ptb": ptb_detokenizer,
    "default": lambda x: x.strip(),
}


def detokenize(text, style="default"):
    """
    Apply detokenization to text.

    Args:
        text: str
        style: str, one of "wikitext", "ptb", "default"

    Returns:
        str, detokenized text
    """
    fn = _DETOKENIZERS.get(style, _DETOKENIZERS["default"])
    return fn(text)


# --- Preprocessing pipeline ---

def preprocess_texts(texts, max_length=512, lowercase=False,
                     strip_newlines=True, min_words=3):
    """
    Preprocess a list of texts for LSME editing.

    Args:
        texts: list of str
        max_length: int, max character length (rough filter before tokenization)
        lowercase: bool
        strip_newlines: bool
        min_words: int, minimum word count to keep

    Returns:
        list of str, filtered and cleaned texts
    """
    cleaned = []
    for text in texts:
        if strip_newlines:
            text = text.replace("\n", " ").replace("\r", " ")
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        if lowercase:
            text = text.lower()
        # Filter too short or too long
        if len(text.split()) < min_words:
            continue
        if len(text) > max_length * 4:  # rough char-level filter
            text = text[:max_length * 4]
        cleaned.append(text)
    return cleaned


def truncate_and_pad(token_ids, max_length, pad_token_id, mask_token_id=None):
    """
    Truncate or pad a 1D list of token IDs to max_length.

    Args:
        token_ids: list of int
        max_length: int
        pad_token_id: int
        mask_token_id: int or None (if provided, pads with MASK for diffusion)

    Returns:
        list of int, length == max_length
    """
    pad_id = mask_token_id if mask_token_id is not None else pad_token_id
    if len(token_ids) >= max_length:
        return token_ids[:max_length]
    return token_ids + [pad_id] * (max_length - len(token_ids))
