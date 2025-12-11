import tiktoken
import torch



class BPETokenizer:
    """
    BPE tokenizer using gpt2 encoding (standard GPT-2 tokenizer).
    
    gpt2 has a vocabulary of 50,257 tokens.
    """
    
    def __init__(self):
        """Initialize the tokenizer with gpt2 encoding."""
        self.encoding = tiktoken.get_encoding("gpt2")
        self._vocab_size = self.encoding.n_vocab
        
        # Special tokens
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = self.encoding.encode(
            self.eos_token, allowed_special={self.eos_token}
        )[0]
        self.pad_token_id = self.eos_token_id
        
    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self._vocab_size
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
    ) -> list[int] | torch.Tensor:
        """
        Encode text to token IDs with optional truncation, padding, and tensor conversion.
        
        Args:
            text: The text to encode.
            add_special_tokens: If True, adds EOS token at the end.
            truncation: If True, truncate to max_length.
            max_length: Maximum sequence length (required if truncation=True or padding="max_length").
            padding: False (no padding), True/"longest", or "max_length".
            return_tensors: None for list, "pt" for PyTorch tensor.
        
        Returns:
            List of token IDs or PyTorch tensor.
        """
        tokens = self.encoding.encode(text)
        
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        
        # Convert to tensor
        if return_tensors == "pt":
            return torch.tensor(tokens, dtype=torch.long)
            
        return tokens
    
    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs to decode.
            skip_special_tokens: If True, removes special tokens from output.
        
        Returns:
            Decoded text string.
        """
        text = self.encoding.decode(token_ids)
        
        if skip_special_tokens:
            text = text.replace(self.eos_token, "")
            
        return text
    
    def encode_batch(self, texts: list[str], add_special_tokens: bool = False) -> list[list[int]]:
        """Encode multiple texts to token IDs."""
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def decode_batch(self, token_id_batches: list[list[int]], skip_special_tokens: bool = False) -> list[str]:
        """Decode multiple token ID sequences back to text."""
        return [self.decode(ids, skip_special_tokens) for ids in token_id_batches]
    
    def __call__(self, text: str | list[str], add_special_tokens: bool = False) -> list[int] | list[list[int]]:
        """Encode text(s) to token IDs. Callable interface."""
        if isinstance(text, str):
            return self.encode(text, add_special_tokens)
        return self.encode_batch(text, add_special_tokens)
    
    def __len__(self) -> int:
        """Return the vocabulary size."""
        return self._vocab_size
    
    def __repr__(self) -> str:
        return f"Tokenizer(vocab_size={self._vocab_size}, encoding=gpt2)"
