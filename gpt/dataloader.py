import os

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import GPT2TokenizerFast

os.environ["TOKENIZER_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"


def get_tokenizer():
    """Load GPT-2 tokenizer with padding token configured."""
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class ChunkedTextDataset(IterableDataset):
    """
    Streaming dataset that tokenizes text and yields fixed-length chunks.
    
    Uses concatenate + chunk strategy: tokenizes all text, concatenates tokens
    with EOS separators, then splits into fixed-length sequences.
    No truncation waste, no padding needed.
    """
    
    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 1024,
        shuffle_buffer: int = 10_000,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle_buffer = shuffle_buffer
    
    def __iter__(self):
        # Load streaming dataset
        ds = load_dataset(
            "nvidia/Nemotron-Pretraining-Dataset-sample",
            "Nemotron-CC-High-Quality",
            streaming=True,
            split="train",
        )
        ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=42)
        
        # Buffer to accumulate tokens
        buffer = []
        
        for example in ds:
            # Tokenize and add EOS token to separate documents
            tokens = self.tokenizer.encode(example["text"])
            tokens.append(self.tokenizer.eos_token_id)
            buffer.extend(tokens)
            
            # Yield complete chunks
            while len(buffer) >= self.max_seq_len:
                chunk = buffer[:self.max_seq_len]
                buffer = buffer[self.max_seq_len:]
                
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids": input_ids,
                    "labels": input_ids.clone(),
                }


def create_dataloader(
    batch_size: int = 8,
    max_seq_len: int = 1024,
    shuffle_buffer: int = 10_000,
):
    """
    Create a PyTorch DataLoader from a streaming HuggingFace dataset.
    
    Args:
        batch_size: Number of samples per batch
        max_seq_len: Maximum sequence length for tokenization
        shuffle_buffer: Buffer size for shuffling the streaming dataset
    
    Returns:
        DataLoader yielding batches of tokenized text
    """
    tokenizer = get_tokenizer()
    
    dataset = ChunkedTextDataset(
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        shuffle_buffer=shuffle_buffer,
    )
    
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "labels": labels}
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # Streaming datasets work best with 0 workers
    )
    
    return dataloader, tokenizer


if __name__ == "__main__":
    # Test the dataloader
    dataloader, tokenizer = create_dataloader(batch_size=4, max_seq_len=128)
    
    for i, batch in enumerate(dataloader):
        print(f"Batch {i}:")
        print(f"  input_ids shape: {batch['input_ids'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  Sample decoded: {tokenizer.decode(batch['input_ids'][0][:50])}...")
        
        if i >= 2:  # Just show a few batches
            break

