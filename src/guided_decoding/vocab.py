"""Bridges token ids to raw bytes, for guides that reason about strings."""

from transformers import PreTrainedTokenizerBase
from transformers.convert_slow_tokenizer import bytes_to_unicode

_BYTE_DECODER = {ch: b for b, ch in bytes_to_unicode().items()}


class _TrieNode:
    __slots__: tuple[str, ...] = ("children", "token_id")

    def __init__(self):
        self.children: dict[int, "_TrieNode"] = {}
        self.token_id: int | None = None


class Vocab:
    """Token ids <-> raw bytes, plus prefix lookups backed by a trie."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, vocab_size: int):
        # vocab_size is the model's logit width; it exceeds len(tokenizer) when the lm_head
        # is padded (e.g. Qwen2.5), so the trailing ids stay None.
        self.token_bytes: list[bytes | None] = [None] * vocab_size
        self.eos_id = tokenizer.eos_token_id
        special_ids = set(tokenizer.all_special_ids) | set(tokenizer.added_tokens_decoder)

        self._root = _TrieNode()
        for token_id in range(len(tokenizer)):
            if token_id in special_ids:
                continue
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token is None:
                continue
            if any(ch not in _BYTE_DECODER for ch in token):
                raise ValueError(
                    f"token {token!r} isn't byte-level BPE (looks like SentencePiece or similar); "
                    + "v1 only supports byte-level tokenizers."
                )
            raw = bytes(_BYTE_DECODER[ch] for ch in token)
            self.token_bytes[token_id] = raw
            self._insert(raw, token_id)

    def _insert(self, raw: bytes, token_id: int) -> None:
        node = self._root
        for b in raw:
            node = node.children.setdefault(b, _TrieNode())
        node.token_id = token_id

    def tokens_prefixing(self, target: bytes) -> list[int]:
        """All token ids whose byte string is a non-empty prefix of `target`."""
        node = self._root
        ids = []
        for b in target:
            node = node.children.get(b)
            if node is None:
                break
            if node.token_id is not None:
                ids.append(node.token_id)
        return ids
