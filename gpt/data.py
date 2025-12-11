# from huggingface_hub import login
# login(token=os.environ["HF_TOKEN"])
import os
from datasets import load_dataset
from tokenizer import BPETokenizer
from transformers import AutoTokenizer

os.environ["TOKENIZER_PARALLELISM"] = "false"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

def load_hf_tokenizer(model_name: str = "gpt2"):
    """
    Load a HuggingFace tokenizer with proper padding configuration.
    
    Args:
        model_name: HuggingFace model name (default: "gpt2")
    
    Returns:
        Configured HuggingFace tokenizer with padding support.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-2 doesn't have a pad token by default, use eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Suppress "sequence longer than model max length" warning
    # We handle chunking ourselves, so this warning is not relevant
    tokenizer.model_max_length = int(1e30)
    
    return tokenizer

# Let's start with a sample for now
def load_hf_dataset(
    path: str = "nvidia/Nemotron-Pretraining-Dataset-sample",
    name: str = "Nemotron-CC-High-Quality",
    streaming = True,
    shuffle_buffer = 10000,
    max_seq_length=1024
):
    ds = load_dataset(
        path=path, 
        name=name, 
        streaming=streaming, 
        split='train'
    )

    # The shuffle_buffer indicates how "random" your dataset gets mixed.
    ds = ds.shuffle(buffer_size=shuffle_buffer, seed=42)
    empty_string = ''
    for data in ds['text']:
        empty_string +=" " + data
    print(len(empty_string) // 1024)
    # Now we need to create the input / target for LLMs
    tokenizer = load_hf_tokenizer("gpt2")
    


if __name__ == '__main__':
    load_hf_dataset()