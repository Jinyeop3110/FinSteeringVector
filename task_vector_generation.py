"""
Task Vector Generation for FinQA.

Extracts hidden representations at specified token positions and layers,
saves per-sample raw representations and computes averaged task vectors.

Usage:
    python task_vector_generation.py --config task_vector_config.yaml
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.datasets import FinQADataset
from src.prompts import VanillaPrompt, FewShotPrompt, ChainOfThoughtPrompt

logger = logging.getLogger(__name__)

PROMPT_TYPES = {
    "vanilla": VanillaPrompt,
    "few_shot": FewShotPrompt,
    "cot": ChainOfThoughtPrompt,
}


def setup_logging(output_dir: Path) -> None:
    """Setup logging to both console and file."""
    log_file = output_dir / "extraction.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def parse_token_positions(positions: list) -> list:
    """
    Parse token position specifications.

    Args:
        positions: List of position specs:
            - "last": Last token of sequence
            - "first": First token
            - integers: Specific token indices (negative for from end)
            - dict with "search": String to search for (finds last occurrence)
              e.g., {"search": "Reasoning:", "offset": 0}

    Returns:
        List of position specs
    """
    parsed = []
    for pos in positions:
        if isinstance(pos, str):
            if pos not in ["last", "first"]:
                raise ValueError(f"Unknown token position: {pos}")
            parsed.append(pos)
        elif isinstance(pos, int):
            parsed.append(pos)
        elif isinstance(pos, dict):
            # Search-based position: {"search": "Reasoning:", "offset": 0}
            if "search" not in pos:
                raise ValueError(f"Search-based position must have 'search' key: {pos}")
            parsed.append(pos)
        else:
            raise ValueError(f"Invalid token position type: {type(pos)}")
    return parsed


def find_last_occurrence_token_idx(
    tokenizer,
    input_ids: torch.Tensor,
    search_string: str,
    offset: int = 0,
) -> int | None:
    """
    Find the token index of the last occurrence of a search string.

    Args:
        tokenizer: HuggingFace tokenizer
        input_ids: Token IDs tensor (1D or 2D with batch=1)
        search_string: String to search for (e.g., "Reasoning:")
        offset: Offset from the start of the found string (0 = first token of string)

    Returns:
        Token index of the last occurrence, or None if not found
    """
    if input_ids.dim() == 2:
        input_ids = input_ids[0]

    # Decode the full text
    full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # Find the last occurrence of the search string in the text
    last_pos = full_text.rfind(search_string)
    if last_pos == -1:
        return None

    # Find which token this character position corresponds to
    # We do this by decoding progressively and finding where the position falls
    cumulative_len = 0
    for token_idx in range(len(input_ids)):
        token_text = tokenizer.decode(input_ids[:token_idx + 1], skip_special_tokens=False)
        if len(token_text) > last_pos:
            # This token contains or is after the start of search_string
            # Apply offset
            result_idx = token_idx + offset
            if result_idx < 0:
                result_idx = 0
            if result_idx >= len(input_ids):
                result_idx = len(input_ids) - 1
            return result_idx

    return None


def get_token_indices_with_search(
    position_specs: list,
    seq_len: int,
    tokenizer=None,
    input_ids: torch.Tensor = None,
) -> tuple[list[int], list[str]]:
    """
    Convert position specifications to actual token indices.
    Supports search-based positions.

    Args:
        position_specs: List of position specs
        seq_len: Sequence length
        tokenizer: HuggingFace tokenizer (needed for search)
        input_ids: Token IDs (needed for search)

    Returns:
        Tuple of (list of token indices, list of position names)
    """
    indices = []
    names = []

    for spec in position_specs:
        if spec == "last":
            indices.append(seq_len - 1)
            names.append("last")
        elif spec == "first":
            indices.append(0)
            names.append("first")
        elif isinstance(spec, int):
            if spec < 0:
                indices.append(seq_len + spec)
            else:
                indices.append(spec)
            names.append(f"idx{spec}")
        elif isinstance(spec, dict):
            # Search-based position
            search_str = spec["search"]
            offset = spec.get("offset", 0)

            if tokenizer is None or input_ids is None:
                raise ValueError("tokenizer and input_ids required for search-based positions")

            token_idx = find_last_occurrence_token_idx(
                tokenizer, input_ids, search_str, offset
            )

            if token_idx is None:
                logger.warning(f"Search string '{search_str}' not found, using last token")
                token_idx = seq_len - 1

            indices.append(token_idx)
            # Clean name for the position
            clean_name = search_str.replace(":", "").replace(" ", "_").lower()
            names.append(f"search_{clean_name}")
        else:
            raise ValueError(f"Unknown position spec: {spec}")

    return indices, names


def get_session_name(model: str, tag: str | None = None) -> str:
    """Generate session folder name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("/")[-1]

    parts = [model_short, "task_vector", timestamp]
    if tag:
        parts.insert(0, tag)
    return "_".join(parts)


class HiddenStateExtractor:
    """Extracts hidden states from transformer models."""

    def __init__(
        self,
        model_name: str,
        layers: list[int],
        token_positions: list,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        """
        Initialize the extractor.

        Args:
            model_name: HuggingFace model name
            layers: List of layer indices to extract from
            token_positions: List of token position specs
            device: Device to use
            dtype: Data type for model
        """
        self.model_name = model_name
        self.layers = sorted(layers)
        self.token_positions = parse_token_positions(token_positions)
        self.device = device
        self.dtype = dtype

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            output_hidden_states=True,
        )
        self.model.eval()

        # Get model config
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        logger.info(f"Model loaded: {self.num_layers} layers, {self.hidden_size} hidden size")
        logger.info(f"Extracting from layers: {self.layers}")
        logger.info(f"Token positions: {self.token_positions}")

    def extract_single(self, prompt: str | list[dict]) -> dict[str, torch.Tensor]:
        """
        Extract hidden states for a single prompt.

        Args:
            prompt: Text prompt or chat format

        Returns:
            Dictionary mapping "layer_{i}_pos_{j}" to hidden state tensors
        """
        # Format prompt if chat format
        if isinstance(prompt, list):
            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        ).to(self.device)

        seq_len = inputs.input_ids.shape[1]

        # Get token indices (supports search-based positions)
        token_indices, pos_names = get_token_indices_with_search(
            self.token_positions,
            seq_len,
            tokenizer=self.tokenizer,
            input_ids=inputs.input_ids,
        )

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Extract hidden states at specified layers and positions
        hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden)

        results = {}
        for layer_idx in self.layers:
            # hidden_states[0] is embeddings, hidden_states[1] is layer 0, etc.
            layer_hidden = hidden_states[layer_idx + 1]  # +1 for embedding layer offset

            for pos_name, token_idx in zip(pos_names, token_indices):
                # Get hidden state at this position
                hidden = layer_hidden[0, token_idx, :].cpu()  # (hidden_size,)

                # Create key
                key = f"layer_{layer_idx}_pos_{pos_name}"
                results[key] = hidden

        # Also store metadata
        results["_seq_len"] = seq_len
        results["_token_indices"] = token_indices
        results["_pos_names"] = pos_names

        return results

    def extract_batch(
        self,
        prompts: list[str | list[dict]],
        show_progress: bool = True,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Extract hidden states for a batch of prompts.

        Note: Due to variable sequence lengths, we process one at a time.
        """
        results = []
        iterator = tqdm(prompts, desc="Extracting") if show_progress else prompts

        for prompt in iterator:
            result = self.extract_single(prompt)
            results.append(result)

        return results


def compute_average_representations(
    all_representations: list[dict[str, torch.Tensor]],
    layers: list[int],
    token_positions: list,
) -> dict[str, torch.Tensor]:
    """
    Compute average representations across all samples.

    Args:
        all_representations: List of per-sample representation dicts
        layers: Layer indices
        token_positions: Token position specs

    Returns:
        Dictionary mapping keys to averaged tensors
    """
    # Collect all tensors for each key
    key_tensors = {}

    for sample_repr in all_representations:
        for key, tensor in sample_repr.items():
            if key.startswith("_"):
                continue  # Skip metadata

            if key not in key_tensors:
                key_tensors[key] = []
            key_tensors[key].append(tensor)

    # Compute averages
    averages = {}
    for key, tensors in key_tensors.items():
        stacked = torch.stack(tensors, dim=0)  # (n_samples, hidden_size)
        averages[key] = stacked.mean(dim=0)    # (hidden_size,)
        averages[f"{key}_std"] = stacked.std(dim=0)  # Also save std

    return averages


def main():
    parser = argparse.ArgumentParser(description="Task Vector Generation for FinQA")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override seed for ICL example selection")
    parser.add_argument("--tag", type=str, default=None, help="Override tag for session naming")
    args = parser.parse_args()

    # Load main config
    config = load_config(args.config)

    # Load prompt config
    prompt_config_path = config.get("prompt_config")
    if prompt_config_path:
        prompt_config = load_config(prompt_config_path)
    else:
        prompt_config = {}

    # Merge configs (main config overrides prompt config for shared keys)
    for key in ["model", "seed", "max_samples", "split"]:
        if key not in config and key in prompt_config:
            config[key] = prompt_config[key]

    # Command line overrides
    if args.seed is not None:
        config["seed"] = args.seed
    if args.tag is not None:
        config["tag"] = args.tag

    # Create output directories
    base_dir = Path(__file__).parent
    task_vector_dir = base_dir / config.get("task_vector_dir", "task_vector")
    raw_repr_dir = base_dir / config.get("raw_representations_dir", "raw_representations")

    session_name = get_session_name(config["model"], config.get("tag"))

    task_vector_session_dir = task_vector_dir / session_name
    raw_repr_session_dir = raw_repr_dir / session_name

    task_vector_session_dir.mkdir(parents=True, exist_ok=True)
    raw_repr_session_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(task_vector_session_dir)

    logger.info("=" * 60)
    logger.info("Task Vector Generation")
    logger.info("=" * 60)
    logger.info(f"Session: {session_name}")
    logger.info(f"Config: {config}")

    # Save config
    with open(task_vector_session_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Initialize prompt template
    prompt_type = prompt_config.get("prompt_type", "vanilla")
    prompt_class = PROMPT_TYPES[prompt_type]
    prompt_kwargs = {}

    if prompt_type == "few_shot":
        prompt_kwargs["n_shots"] = prompt_config.get("n_shots", 3)
        prompt_kwargs["output_program"] = prompt_config.get("output_program", False)
    elif prompt_type == "cot":
        n_shots = prompt_config.get("n_shots", 0)
        prompt_kwargs["n_shots"] = n_shots if n_shots > 0 else 2
        if prompt_config.get("no_context_in_examples", False):
            prompt_kwargs["include_context_in_examples"] = False
        if prompt_config.get("table_only_in_examples", False):
            prompt_kwargs["table_only_in_examples"] = True

    prompt_template = prompt_class(**prompt_kwargs)
    logger.info(f"Using prompt type: {prompt_type}")

    # Load dataset
    split = config.get("split", "train")
    seed = config.get("seed", 42)
    max_samples = config.get("max_samples")  # None means all samples
    use_qa_json = prompt_config.get("use_qa_json", False) or config.get("use_qa_json", False)

    dataset = FinQADataset(split=split, seed=seed, use_qa_json=use_qa_json)
    dataset.load()
    logger.info(f"Loaded {split} dataset with {len(dataset)} examples (use_qa_json={use_qa_json})")

    # Load ICL pool if needed
    icl_pool = None
    n_shots = prompt_config.get("n_shots", 0)
    if prompt_type in ["few_shot", "cot"] and n_shots > 0:
        icl_pool = FinQADataset(split="train", seed=seed, use_qa_json=use_qa_json)
        icl_pool.load()
        logger.info(f"Loaded train dataset with {len(icl_pool)} examples for ICL")

    # Prepare prompts
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    prompts = []
    sample_ids = []

    logger.info(f"Preparing {n_samples} prompts...")
    for idx in tqdm(range(n_samples), desc="Preparing prompts"):
        example = dataset[idx]
        question = example["question"]
        context = example["context"]

        # Get ICL examples if needed
        icl_examples = None
        if n_shots > 0 and icl_pool is not None:
            icl_examples = dataset.get_icl_examples(n_shots, idx, icl_pool=icl_pool)

        # Format prompt
        prompt = prompt_template.format(
            question=question,
            context=context,
            icl_examples=icl_examples,
        )

        prompts.append(prompt)
        sample_ids.append(example.get("id", f"{split}_{idx}"))

    # Initialize extractor
    layers = config.get("layers", [4, 8, 12, 16, 20, 24, 27])
    token_positions = config.get("token_positions", ["last"])

    extractor = HiddenStateExtractor(
        model_name=config["model"],
        layers=layers,
        token_positions=token_positions,
    )

    # Extract representations
    logger.info("Extracting hidden representations...")
    all_representations = extractor.extract_batch(prompts)

    # Save per-sample representations
    logger.info("Saving per-sample representations...")
    for idx, (sample_id, repr_dict) in enumerate(zip(sample_ids, all_representations)):
        # Save as .pt file
        sample_file = raw_repr_session_dir / f"sample_{idx:05d}.pt"

        # Add sample metadata
        repr_dict["_sample_id"] = sample_id
        repr_dict["_sample_idx"] = idx

        torch.save(repr_dict, sample_file)

    logger.info(f"Saved {len(all_representations)} sample representations to {raw_repr_session_dir}")

    # Compute and save averaged task vectors
    logger.info("Computing averaged task vectors...")
    avg_representations = compute_average_representations(
        all_representations, layers, token_positions
    )

    # Save task vectors
    task_vector_file = task_vector_session_dir / "task_vectors.pt"
    torch.save(avg_representations, task_vector_file)
    logger.info(f"Saved task vectors to {task_vector_file}")

    # Also save as individual layer files for convenience
    for layer_idx in layers:
        layer_data = {}
        for key, tensor in avg_representations.items():
            if f"layer_{layer_idx}_" in key:
                layer_data[key] = tensor

        layer_file = task_vector_session_dir / f"layer_{layer_idx}.pt"
        torch.save(layer_data, layer_file)

    # Save metadata
    metadata = {
        "session_name": session_name,
        "model": config["model"],
        "n_samples": n_samples,
        "layers": layers,
        "token_positions": token_positions,
        "prompt_type": prompt_type,
        "prompt_config": prompt_config_path,
        "split": split,
        "hidden_size": extractor.hidden_size,
        "sample_ids": sample_ids,
    }

    with open(task_vector_session_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Task vectors saved to: {task_vector_session_dir}")
    logger.info(f"Raw representations saved to: {raw_repr_session_dir}")
    logger.info(f"Layers extracted: {layers}")
    logger.info(f"Token positions: {token_positions}")
    logger.info(f"Samples processed: {n_samples}")

    # Print shape info
    for key, tensor in avg_representations.items():
        if not key.endswith("_std"):
            logger.info(f"  {key}: {tensor.shape}")


if __name__ == "__main__":
    main()
