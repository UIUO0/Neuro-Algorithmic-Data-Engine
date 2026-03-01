"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Neuro-Algorithmic-Data-Engine — Data Loader Module                        ║
║                                                                            ║
║  Text-to-SQL Pipeline: Streaming Data Loader                               ║
║  Transforms natural language questions into complex SQL queries.            ║
║  Designed for Big Data processing within 16GB Unified Memory constraints.  ║
║                                                                            ║
║  Author: Neuro-Algorithmic-Data-Engine Team                                ║
║  License: MIT                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Strategic Direction:
────────────────────
The core Algorithmic Application of this engine is Text-to-SQL — translating
natural language questions into complex, executable SQL queries. This represents
a high-value intersection of NLP and Data Analysis:

  • NLP Component: Understanding user intent from unstructured natural language
  • Data Analysis Component: Generating structurally valid SQL that operates
    on real-world database schemas (JOINs, subqueries, aggregations, etc.)
  • Big Data Relevance: Enabling non-technical users to query massive databases
    without writing SQL manually — a critical need in enterprise analytics.

Data Engineering Pipeline:
──────────────────────────
Source: b-mc2/sql-create-context (HuggingFace)

  ┌─────────────────────────────────────────────────────────────┐
  │  Raw Dataset Columns                                        │
  │  ├── question : Natural language question (NL input)        │
  │  ├── context  : Table schema — CREATE TABLE statements      │
  │  └── answer   : Target SQL query (ground truth)             │
  ├─────────────────────────────────────────────────────────────┤
  │  Data Processing (this module)                              │
  │  ├── INPUT  = question + context → merged structured prompt │
  │  │           "### Table Schema:\n...\n### Question:\n..."   │
  │  └── OUTPUT = answer → SQL query as training target         │
  ├─────────────────────────────────────────────────────────────┤
  │  Export Format (mlx_lm.lora compatible)                     │
  │  ├── Chat:        {"messages": [{system}, {user}, {asst}]}  │
  │  └── Completions: {"prompt": "...", "completion": "..."}    │
  └─────────────────────────────────────────────────────────────┘

Architecture Overview:
─────────────────────
This module implements a memory-efficient data pipeline using HuggingFace's
streaming API. Instead of loading entire datasets into RAM (which would be
impossible for multi-GB datasets on a 16GB machine), we process data as a
lazy stream — one example at a time.

This is critical for Big Data scenarios: the sql-create-context dataset
contains ~78K examples, and future SQL corpora can scale to millions.
The streaming architecture ensures the pipeline handles any scale without
modification — only wall-clock time changes, not memory usage.

Memory Model:
─────────────
┌─────────────────────────────────────────────────────┐
│  16GB Unified Memory Budget                         │
│  ├── ~4-5 GB  → Quantized Model (4-bit, 8B params) │
│  ├── ~2-3 GB  → Gradients + Optimizer States        │
│  ├── ~1-2 GB  → Activations (batch_size=1)          │
│  ├── ~1   GB  → OS + System Overhead                │
│  └── ~1-2 GB  → Data Pipeline (this module) ←       │
│  Remaining  → Safety Buffer                         │
└─────────────────────────────────────────────────────┘

Key Design Principles:
  1. Streaming-first: O(1) memory regardless of dataset size
  2. Generator-based: Uses Python generators to avoid materialization
  3. Memory monitoring: Real-time psutil checks with safety thresholds
  4. Batched I/O: Writes to disk in configurable chunks to minimize I/O calls
  5. SQL-aware formatting: Structured prompts that preserve table schema context
"""

import json
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, Optional, Any

import psutil
from tqdm import tqdm
from datasets import load_dataset

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Memory safety threshold: stop processing if available RAM drops below this (GB)
MEMORY_SAFETY_THRESHOLD_GB = 2.0

# Default maximum sequence length (tokens) — shorter = less memory per sample
DEFAULT_MAX_SEQ_LENGTH = 512

# Batch size for writing to disk (number of samples per I/O flush)
DISK_WRITE_BATCH_SIZE = 100

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for the data loading pipeline.

    Target Dataset: b-mc2/sql-create-context
    ─────────────────────────────────────────
    This dataset contains natural language questions paired with SQL table
    schemas (CREATE TABLE statements) and their corresponding SQL queries.

    Dataset columns:
      - question : The user's natural language question
      - context  : SQL table schema (CREATE TABLE statements)
      - answer   : The target SQL query

    Data Engineering:
      Input (User Prompt)  = question + context (merged)
      Output (Target)      = answer (SQL query)

    Attributes:
        dataset_name: HuggingFace dataset identifier.
        dataset_config: Optional dataset configuration/subset name.
        split: Dataset split to load (default: 'train').
        input_field: Name of the question/prompt field in the dataset.
        context_field: Name of the context/schema field in the dataset.
        output_field: Name of the output/answer field in the dataset.
        system_prompt: Optional system prompt to prepend to every example.
        max_seq_length: Maximum token sequence length for truncation.
        output_dir: Directory to save processed JSONL files.
        train_ratio: Fraction of data for training (default: 0.85).
        valid_ratio: Fraction of data for validation (default: 0.10).
        test_ratio: Fraction of data for testing (default: 0.05).
        max_samples: Optional cap on total samples to process (None = all).
        streaming: Whether to use streaming mode (must be True for Big Data).
    """
    dataset_name: str = "b-mc2/sql-create-context"
    dataset_config: Optional[str] = None
    split: str = "train"
    input_field: str = "question"
    context_field: str = "context"
    output_field: str = "answer"
    system_prompt: Optional[str] = (
        "You are a SQL expert. Given a user question and the relevant table "
        "schemas, generate the correct SQL query to answer the question."
    )
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH
    output_dir: str = "data"
    train_ratio: float = 0.85
    valid_ratio: float = 0.10
    test_ratio: float = 0.05
    max_samples: Optional[int] = None
    streaming: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Memory Monitoring Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_available_memory_gb() -> float:
    """Return available system memory in gigabytes.

    Complexity:
        Time:  O(1) — single system call via psutil
        Space: O(1) — returns a single float value

    Returns:
        Available memory in GB.
    """
    return psutil.virtual_memory().available / (1024 ** 3)


def check_memory_safety(threshold_gb: float = MEMORY_SAFETY_THRESHOLD_GB) -> bool:
    """Check if available memory is above the safety threshold.

    This is called periodically during data processing to prevent OOM crashes.
    On a 16GB machine, we want at least 2GB free as a safety buffer.

    Complexity:
        Time:  O(1) — single system call
        Space: O(1) — boolean return

    Args:
        threshold_gb: Minimum available memory required (in GB).

    Returns:
        True if memory is safe, False if approaching dangerous levels.
    """
    available = get_available_memory_gb()
    if available < threshold_gb:
        logger.warning(
            f"⚠ Low memory: {available:.2f} GB available "
            f"(threshold: {threshold_gb:.2f} GB)"
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Core Data Loader
# ─────────────────────────────────────────────────────────────────────────────

class NeurDataLoader:
    """Memory-efficient streaming data loader for HuggingFace datasets.

    This class provides a pipeline for:
      1. Streaming data from HuggingFace Hub (no full download required)
      2. Transforming raw examples into chat/completions format
      3. Merging question + context fields into structured prompts
      4. Filtering and validating samples
      5. Exporting to JSONL files compatible with mlx_lm.lora

    The entire pipeline operates in O(1) memory relative to dataset size
    by leveraging Python generators and HuggingFace's streaming mode.

    Target Dataset: b-mc2/sql-create-context
        Input  = question (NL question) + context (table schema)
        Output = answer (SQL query)

    Example Usage:
        >>> config = DataConfig()  # defaults to b-mc2/sql-create-context
        >>> loader = NeurDataLoader(config)
        >>> loader.prepare_dataset()  # Streams, transforms, and saves to disk
    """

    def __init__(self, config: DataConfig):
        """Initialize the data loader.

        Complexity:
            Time:  O(1) — only stores configuration, no data loaded
            Space: O(1) — only the config dataclass is stored

        Args:
            config: DataConfig instance with all pipeline settings.
        """
        self.config = config
        self._validate_config()

        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        logger.info(f"NeurDataLoader initialized for: {config.dataset_name}")
        logger.info(f"Output directory: {config.output_dir}")

    def _validate_config(self) -> None:
        """Validate configuration parameters.

        Complexity:
            Time:  O(1) — simple arithmetic checks
            Space: O(1) — no allocations

        Raises:
            ValueError: If ratios don't sum to 1.0 or are out of range.
        """
        total_ratio = (
            self.config.train_ratio
            + self.config.valid_ratio
            + self.config.test_ratio
        )
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total_ratio:.2f} "
                f"(train={self.config.train_ratio}, "
                f"valid={self.config.valid_ratio}, "
                f"test={self.config.test_ratio})"
            )

        if self.config.max_seq_length < 1:
            raise ValueError(
                f"max_seq_length must be >= 1, got {self.config.max_seq_length}"
            )

    def _stream_raw_dataset(self) -> Iterator[dict]:
        """Stream raw examples from HuggingFace Hub.

        This is the entry point of the data pipeline. It creates a lazy
        iterator over the dataset — no data is downloaded until iteration
        begins, and at most ONE example is in memory at a time.

        Complexity:
            Time:  O(N) total across full iteration, where N = dataset size
                   O(1) per yielded example (streaming fetch + yield)
            Space: O(1) — only one example dict in memory at any time
                   The HuggingFace streaming client maintains a small internal
                   buffer (~few KB) for HTTP chunked transfer.

        Memory Impact (16GB budget):
            ~10-50 KB per example (text + metadata)
            Total: O(1) regardless of dataset size ✓

        Yields:
            Raw example dicts from the HuggingFace dataset.
        """
        logger.info(f"Initializing stream from: {self.config.dataset_name}")

        dataset_kwargs: dict[str, Any] = {
            "path": self.config.dataset_name,
            "split": self.config.split,
            "streaming": self.config.streaming,
        }
        if self.config.dataset_config:
            dataset_kwargs["name"] = self.config.dataset_config

        dataset = load_dataset(**dataset_kwargs)

        count = 0
        for example in dataset:
            # Periodic memory safety check (every 1000 samples to avoid
            # overhead from frequent system calls)
            if count % 1000 == 0 and not check_memory_safety():
                logger.error(
                    f"Memory safety threshold breached after {count} samples. "
                    f"Stopping stream to prevent OOM."
                )
                break

            count += 1

            # Respect max_samples cap if configured
            if self.config.max_samples and count > self.config.max_samples:
                logger.info(
                    f"Reached max_samples limit: {self.config.max_samples}"
                )
                break

            yield example

        logger.info(f"Streamed {count} examples total.")

    def _build_user_prompt(
        self, question: str, context: str
    ) -> str:
        """Merge the question and context fields into a structured user prompt.

        Data Engineering Logic:
            The user prompt is constructed by combining the natural language
            question with the SQL table schema (CREATE TABLE statements).
            This gives the model both the intent and the structural context
            needed to generate the correct SQL query.

        Output Format:
            ### Table Schema:
            CREATE TABLE ...

            ### Question:
            What is the total revenue ...?

        Complexity:
            Time:  O(L) where L = len(question) + len(context) — string concat
            Space: O(L) — one new string allocated per call

        Memory Impact (16GB budget):
            ~1-5 KB per prompt — completely negligible ✓

        Args:
            question: The user's natural language question.
            context: The SQL table schema (CREATE TABLE statements).

        Returns:
            Formatted user prompt string.
        """
        parts = []

        if context:
            parts.append(f"### Table Schema:\n{context}")

        if question:
            parts.append(f"### Question:\n{question}")

        return "\n\n".join(parts)

    def _transform_to_chat_format(
        self, raw_stream: Iterator[dict]
    ) -> Iterator[dict]:
        """Transform raw examples into mlx_lm-compatible chat format.

        Data Engineering for b-mc2/sql-create-context:
            - question + context → merged into a structured user message
            - answer (SQL query)  → assistant response

        Converts each raw example into the 'chat' JSONL format expected
        by mlx_lm.lora:
            {"messages": [
                {"role": "system", "content": "You are a SQL expert..."},
                {"role": "user", "content": "### Table Schema:\n...\n\n### Question:\n..."},
                {"role": "assistant", "content": "SELECT ..."}
            ]}

        This is a pure streaming transformation — each input example
        produces exactly one output example with no buffering.

        Complexity:
            Time:  O(N) total, O(1) per example — simple dict construction
                   + O(L) string concat per example where L = text length
            Space: O(1) — transforms in-place, one example at a time
                   Each transformed dict is ~same size as input (~1-5 KB)

        Memory Impact (16GB budget):
            Negligible — same as raw stream, no amplification ✓

        Args:
            raw_stream: Iterator of raw example dicts.

        Yields:
            Transformed dicts in mlx_lm chat format.
        """
        for example in raw_stream:
            # Extract fields: question, context (schema), and answer (SQL)
            question = str(example.get(self.config.input_field, "")).strip()
            context = str(example.get(self.config.context_field, "")).strip()
            answer = str(example.get(self.config.output_field, "")).strip()

            # Skip examples with missing question or answer
            if not question or not answer:
                continue

            # Build the merged user prompt (question + table schema)
            user_content = self._build_user_prompt(question, context)

            # Build chat messages
            messages = []

            # Add system prompt if configured
            if self.config.system_prompt:
                messages.append({
                    "role": "system",
                    "content": self.config.system_prompt,
                })

            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": answer})

            yield {"messages": messages}

    def _transform_to_completions_format(
        self, raw_stream: Iterator[dict]
    ) -> Iterator[dict]:
        """Transform raw examples into mlx_lm-compatible completions format.

        Data Engineering for b-mc2/sql-create-context:
            - question + context → merged into the prompt
            - answer (SQL query)  → completion

        Output format:
            {"prompt": "### Table Schema:\n...\n\n### Question:\n...",
             "completion": "SELECT ..."}

        Complexity:
            Time:  O(N) total, O(1) per example
            Space: O(1) — one example at a time

        Args:
            raw_stream: Iterator of raw example dicts.

        Yields:
            Transformed dicts in mlx_lm completions format.
        """
        for example in raw_stream:
            question = str(example.get(self.config.input_field, "")).strip()
            context = str(example.get(self.config.context_field, "")).strip()
            answer = str(example.get(self.config.output_field, "")).strip()

            if not question or not answer:
                continue

            prompt = self._build_user_prompt(question, context)

            yield {"prompt": prompt, "completion": answer}

    def _write_jsonl_streaming(
        self,
        data_stream: Iterator[dict],
        filepath: Path,
        description: str = "Writing",
    ) -> int:
        """Write a stream of dicts to a JSONL file with batched I/O.

        Uses buffered writes to minimize disk I/O system calls. Every
        DISK_WRITE_BATCH_SIZE samples, the buffer is flushed to disk.

        Complexity:
            Time:  O(N) — one JSON serialization + write per example
                   I/O flushes occur every DISK_WRITE_BATCH_SIZE samples,
                   so total flushes = O(N / DISK_WRITE_BATCH_SIZE)
            Space: O(B) where B = DISK_WRITE_BATCH_SIZE
                   At most B serialized JSON strings are buffered before flush.
                   With B=100 and ~100 bytes/line, buffer ≈ 10 KB

        Memory Impact (16GB budget):
            ~10 KB buffer — completely negligible ✓

        Args:
            data_stream: Iterator of dicts to serialize.
            filepath: Output file path.
            description: Progress bar description.

        Returns:
            Number of samples written.
        """
        count = 0
        buffer: list[str] = []

        with open(filepath, "w", encoding="utf-8") as f:
            for item in tqdm(data_stream, desc=description, unit=" samples"):
                buffer.append(json.dumps(item, ensure_ascii=False))
                count += 1

                # Flush buffer to disk periodically
                if len(buffer) >= DISK_WRITE_BATCH_SIZE:
                    f.write("\n".join(buffer) + "\n")
                    buffer.clear()

            # Flush remaining items
            if buffer:
                f.write("\n".join(buffer) + "\n")
                buffer.clear()

        logger.info(f"Wrote {count} samples to {filepath}")
        return count

    def prepare_dataset(self, output_format: str = "chat") -> dict[str, int]:
        """Main pipeline: stream → transform → split → save to disk.

        This is the primary entry point. It orchestrates the full pipeline:
          1. Stream data from HuggingFace (no full download)
          2. Transform each example to mlx_lm format
          3. Deterministically split into train/valid/test
          4. Write each split to its own JSONL file

        The splitting uses a simple modular arithmetic approach to avoid
        needing to know the total dataset size upfront (which isn't available
        in streaming mode).

        Complexity:
            Time:  O(N) — single pass through the dataset
                   Each example is: streamed O(1) + transformed O(1) +
                   serialized O(1) + written O(1) = O(1) per example
            Space: O(B) where B = DISK_WRITE_BATCH_SIZE (the I/O buffer)
                   The stream itself is O(1) — only one example in memory.
                   Split files are written incrementally, not accumulated.

        Memory Impact (16GB budget):
            Peak: ~50 KB (current example + I/O buffer + overhead)
            This leaves >14 GB free for the model during training ✓

        Args:
            output_format: Either 'chat' or 'completions'. Determines the
                           output JSONL structure (default: 'chat').

        Returns:
            Dict with counts: {'train': N, 'valid': N, 'test': N}
        """
        logger.info("=" * 60)
        logger.info("Starting data preparation pipeline")
        logger.info(f"  Dataset:   {self.config.dataset_name}")
        logger.info(f"  Format:    {output_format}")
        logger.info(f"  Streaming: {self.config.streaming}")
        logger.info(f"  Max Seq:   {self.config.max_seq_length}")
        logger.info(f"  Memory:    {get_available_memory_gb():.2f} GB available")
        logger.info("=" * 60)

        # Pre-flight memory check
        if not check_memory_safety():
            raise MemoryError(
                "Insufficient memory to start data pipeline. "
                f"Available: {get_available_memory_gb():.2f} GB, "
                f"Required: >{MEMORY_SAFETY_THRESHOLD_GB} GB"
            )

        # Step 1: Create the raw stream (O(1) memory — lazy iterator)
        raw_stream = self._stream_raw_dataset()

        # Step 2: Transform to target format (O(1) memory — lazy)
        if output_format == "chat":
            transformed_stream = self._transform_to_chat_format(raw_stream)
        elif output_format == "completions":
            transformed_stream = self._transform_to_completions_format(raw_stream)
        else:
            raise ValueError(
                f"Unsupported format: '{output_format}'. "
                f"Use 'chat' or 'completions'."
            )

        # Step 3: Split and write to disk
        # We use a deterministic hash-based split to avoid needing dataset
        # size. For each example at index i:
        #   - i % 100 < train_ratio * 100 → train
        #   - i % 100 < (train_ratio + valid_ratio) * 100 → valid
        #   - else → test
        output_dir = Path(self.config.output_dir)
        train_path = output_dir / "train.jsonl"
        valid_path = output_dir / "valid.jsonl"
        test_path = output_dir / "test.jsonl"

        train_threshold = int(self.config.train_ratio * 100)
        valid_threshold = int(
            (self.config.train_ratio + self.config.valid_ratio) * 100
        )

        counts = {"train": 0, "valid": 0, "test": 0}

        with (
            open(train_path, "w", encoding="utf-8") as f_train,
            open(valid_path, "w", encoding="utf-8") as f_valid,
            open(test_path, "w", encoding="utf-8") as f_test,
        ):
            for i, item in enumerate(
                tqdm(transformed_stream, desc="Processing", unit=" samples")
            ):
                line = json.dumps(item, ensure_ascii=False) + "\n"
                bucket = i % 100

                if bucket < train_threshold:
                    f_train.write(line)
                    counts["train"] += 1
                elif bucket < valid_threshold:
                    f_valid.write(line)
                    counts["valid"] += 1
                else:
                    f_test.write(line)
                    counts["test"] += 1

                # Periodic memory check
                if i % 5000 == 0 and i > 0:
                    if not check_memory_safety():
                        logger.warning(
                            f"Memory pressure detected at sample {i}. "
                            f"Stopping early."
                        )
                        break

        # Summary
        logger.info("=" * 60)
        logger.info("Data preparation complete!")
        logger.info(f"  Train: {counts['train']:,} samples → {train_path}")
        logger.info(f"  Valid: {counts['valid']:,} samples → {valid_path}")
        logger.info(f"  Test:  {counts['test']:,} samples  → {test_path}")
        logger.info(f"  Memory: {get_available_memory_gb():.2f} GB available")
        logger.info("=" * 60)

        return counts


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point for the data loader.

    Usage:
        python data_loader.py
        python data_loader.py --dataset "b-mc2/sql-create-context" --format chat
        python data_loader.py --max-samples 5000

    The CLI provides a quick way to run the data pipeline with default
    or custom settings. For programmatic use, instantiate NeurDataLoader
    directly.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Neuro-Algorithmic-Data-Engine: Streaming Data Loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load sql-create-context dataset with default settings (chat format)
  python data_loader.py

  # Load with completions format instead of chat
  python data_loader.py --format completions

  # Limit to 5000 samples for quick testing
  python data_loader.py --max-samples 5000

  # Load a different dataset with custom field mapping
  python data_loader.py --dataset "Open-Orca/OpenOrca" \\
                         --input-field "question" \\
                         --context-field "" \\
                         --output-field "response"
        """,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="b-mc2/sql-create-context",
        help="HuggingFace dataset name (default: b-mc2/sql-create-context)",
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Dataset configuration/subset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split (default: train)",
    )
    parser.add_argument(
        "--input-field",
        type=str,
        default="question",
        help="Input/question field name (default: question)",
    )
    parser.add_argument(
        "--context-field",
        type=str,
        default="context",
        help="Context/schema field name (default: context)",
    )
    parser.add_argument(
        "--output-field",
        type=str,
        default="answer",
        help="Output/answer field name (default: answer)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt to prepend to every example",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LENGTH,
        help=f"Max sequence length in tokens (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for JSONL files (default: data)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["chat", "completions"],
        default="chat",
        help="Output format: 'chat' or 'completions' (default: chat)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.85,
        help="Training split ratio (default: 0.85)",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.10,
        help="Validation split ratio (default: 0.10)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.05,
        help="Test split ratio (default: 0.05)",
    )

    args = parser.parse_args()

    # Build configuration from CLI arguments
    config = DataConfig(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        input_field=args.input_field,
        context_field=args.context_field,
        output_field=args.output_field,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        max_samples=args.max_samples,
        streaming=True,  # Always stream for Big Data safety
    )

    # Run the pipeline
    loader = NeurDataLoader(config)
    counts = loader.prepare_dataset(output_format=args.format)

    # Print summary
    total = sum(counts.values())
    print(f"\n✅ Done! Processed {total:,} samples total.")
    print(f"   Files saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
