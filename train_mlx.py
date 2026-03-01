"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Neuro-Algorithmic-Data-Engine — Training Script                           ║
║                                                                            ║
║  Parameter-Efficient Fine-Tuning (PEFT) via LoRA/QLoRA                     ║
║  Target Model: DeepSeek-R1-Distill-Llama-8B (4-bit quantized)             ║
║  Framework: Apple MLX (mlx-lm)                                             ║
║  Hardware: MacBook M4, 16GB Unified Memory                                 ║
║                                                                            ║
║  Author: Neuro-Algorithmic-Data-Engine Team                                ║
║  License: MIT                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture Overview:
─────────────────────
This script orchestrates the LoRA/QLoRA fine-tuning pipeline using the
mlx_lm.lora official CLI. It handles:
  1. Pre-flight validation (memory, dependencies, data)
  2. YAML configuration generation for mlx_lm.lora
  3. Training execution with optimized hyperparameters
  4. Post-training adapter fusion (optional)
  5. Test generation with the fine-tuned model

Memory Budget for 16GB Unified Memory (QLoRA, 4-bit):
─────────────────────────────────────────────────────
┌─────────────────────────────────────────────────────┐
│  Component                     │  Estimated Usage   │
│  ─────────────────────────────────────────────────  │
│  4-bit Quantized Model (8B)    │  ~4.5 GB           │
│  LoRA Adapters (rank=8)        │  ~0.1 GB           │
│  Optimizer States (AdamW)      │  ~0.2 GB           │
│  Gradients (8 layers)          │  ~0.5 GB           │
│  Activations (batch=1, seq=512)│  ~1.0 GB           │
│  Tokenizer + Overhead          │  ~0.5 GB           │
│  OS + System                   │  ~2.0 GB           │
│  ─────────────────────────────────────────────────  │
│  TOTAL ESTIMATED               │  ~8.8 GB           │
│  SAFETY MARGIN                 │  ~7.2 GB           │
└─────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import subprocess
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml
import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Training Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Complete configuration for LoRA/QLoRA fine-tuning.

    All defaults are tuned for a MacBook M4 with 16GB Unified Memory.
    The 4-bit quantized model enables QLoRA, which dramatically reduces
    memory usage compared to full LoRA on the fp16 model.

    Attributes:
        --- Model ---
        model: HuggingFace model ID or local path. We use the 4-bit quantized
               version from mlx-community for QLoRA compatibility.
        trust_remote_code: Whether to trust remote code in the tokenizer.

        --- LoRA Hyperparameters ---
        fine_tune_type: Type of fine-tuning ('lora', 'dora', or 'full').
        lora_layers: Number of transformer layers to apply LoRA to.
                     Default 16 reduced to 8 for 16GB memory safety.
        lora_rank: Rank of the LoRA decomposition matrices.
                   Lower rank = fewer trainable params = less memory.
        lora_scale: Scaling factor (alpha/rank). Controls adaptation strength.
        lora_dropout: Dropout probability for LoRA layers (0.0 = no dropout).

        --- Training Loop ---
        num_iters: Total number of training iterations.
        batch_size: Samples per gradient step. Set to 1 for memory safety.
        grad_accumulation_steps: Accumulate gradients over N steps before
                                 updating. Effective batch = batch_size * N.
        learning_rate: Peak learning rate for the optimizer.
        optimizer: Optimizer algorithm ('adam', 'adamw', 'sgd', 'adafactor').
        lr_schedule: Learning rate schedule config (dict or None).
                     If None, uses constant learning_rate with optional warmup.

        --- Sequence ---
        max_seq_length: Maximum input sequence length (tokens).
                        Directly impacts memory: longer = more activations.

        --- Regularization ---
        grad_checkpoint: Enable gradient checkpointing to trade compute
                         for memory. Essential for large models on 16GB.
        mask_prompt: If True, only compute loss on the completion tokens,
                     not the prompt tokens. Improves training signal.

        --- I/O ---
        data_dir: Directory containing train.jsonl, valid.jsonl, test.jsonl.
        adapter_path: Directory to save LoRA adapter weights.
        output_dir: Directory for fused model output.
        save_every: Save adapter checkpoint every N iterations.
        val_every: Run validation every N iterations.
        steps_per_report: Log training metrics every N iterations.

        --- Generation ---
        test_prompt: Prompt for test generation after training.
        max_tokens: Maximum tokens to generate during testing.
    """

    # --- Model ---
    model: str = "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit"
    trust_remote_code: bool = True

    # --- LoRA Hyperparameters ---
    fine_tune_type: str = "lora"
    lora_layers: int = 8          # Reduced from default 16 for 16GB safety
    lora_rank: int = 8            # Good balance of capacity vs. memory
    lora_scale: float = 16.0      # alpha = rank * scale = 128
    lora_dropout: float = 0.0     # No dropout for stable QLoRA training

    # --- Training Loop ---
    num_iters: int = 600
    batch_size: int = 1           # Minimum for 16GB memory safety
    grad_accumulation_steps: int = 4  # Effective batch size = 4
    learning_rate: float = 1e-5
    optimizer: str = "adam"       # Adam optimizer (default in mlx-lm)
    lr_schedule: Optional[dict] = None  # None = constant LR (simplest & safest)

    # --- Sequence ---
    max_seq_length: int = 512     # Conservative for memory

    # --- Regularization ---
    grad_checkpoint: bool = True  # Essential for 16GB
    mask_prompt: bool = True      # Better training signal

    # --- I/O ---
    data_dir: str = "data"
    adapter_path: str = "adapters"
    output_dir: str = "fused_model"
    save_every: int = 100
    steps_per_eval: int = 50      # Validation frequency (mlx-lm key name)
    steps_per_report: int = 10
    seed: int = 0

    # --- Generation ---
    test_prompt: str = (
        "### Table Schema:\n"
        "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary DECIMAL);\n"
        "CREATE TABLE departments (id INT, name VARCHAR, budget DECIMAL);\n\n"
        "### Question:\n"
        "What is the average salary for each department?"
    )
    max_tokens: int = 256


# ─────────────────────────────────────────────────────────────────────────────
# Pre-Flight Validation
# ─────────────────────────────────────────────────────────────────────────────

class PreFlightChecker:
    """Validates system readiness before training begins.

    Checks:
      1. Available memory meets minimum requirements
      2. Required Python packages are installed
      3. Training data files exist in the expected format
      4. Output directories are writable
    """

    # Minimum available memory to start training (GB)
    MIN_MEMORY_GB = 6.0

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.checks_passed = 0
        self.checks_failed = 0

    def run_all_checks(self) -> bool:
        """Run all pre-flight checks.

        Returns:
            True if all critical checks pass, False otherwise.
        """
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT CHECKS")
        logger.info("=" * 60)

        results = [
            self._check_memory(),
            self._check_dependencies(),
            self._check_data(),
            self._check_directories(),
        ]

        logger.info("-" * 60)
        logger.info(
            f"Results: {self.checks_passed} passed, "
            f"{self.checks_failed} failed"
        )
        logger.info("=" * 60)

        return all(results)

    def _check_memory(self) -> bool:
        """Check if available memory meets the minimum threshold."""
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        total_gb = psutil.virtual_memory().total / (1024 ** 3)

        if available_gb >= self.MIN_MEMORY_GB:
            logger.info(
                f"  ✅ Memory: {available_gb:.1f} GB available "
                f"/ {total_gb:.1f} GB total"
            )
            self.checks_passed += 1
            return True
        else:
            logger.error(
                f"  ❌ Memory: {available_gb:.1f} GB available "
                f"(need >= {self.MIN_MEMORY_GB} GB). "
                f"Close other applications and retry."
            )
            self.checks_failed += 1
            return False

    def _check_dependencies(self) -> bool:
        """Check that required Python packages are importable."""
        required = ["mlx", "mlx_lm", "transformers", "datasets", "yaml"]
        missing = []

        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)

        if not missing:
            logger.info("  ✅ Dependencies: all required packages found")
            self.checks_passed += 1
            return True
        else:
            logger.error(
                f"  ❌ Dependencies: missing packages: {', '.join(missing)}. "
                f"Run: pip install -r requirements.txt"
            )
            self.checks_failed += 1
            return False

    def _check_data(self) -> bool:
        """Check that training data files exist."""
        data_dir = Path(self.config.data_dir)
        train_file = data_dir / "train.jsonl"

        if train_file.exists():
            # Count lines to verify non-empty
            with open(train_file, "r") as f:
                line_count = sum(1 for _ in f)

            if line_count > 0:
                logger.info(
                    f"  ✅ Data: {train_file} found ({line_count:,} samples)"
                )
                self.checks_passed += 1
                return True
            else:
                logger.error(f"  ❌ Data: {train_file} is empty")
                self.checks_failed += 1
                return False
        else:
            logger.error(
                f"  ❌ Data: {train_file} not found. "
                f"Run data_loader.py first to prepare the dataset."
            )
            self.checks_failed += 1
            return False

    def _check_directories(self) -> bool:
        """Ensure output directories exist and are writable."""
        try:
            os.makedirs(self.config.adapter_path, exist_ok=True)
            os.makedirs(self.config.output_dir, exist_ok=True)
            logger.info(
                f"  ✅ Directories: adapter_path={self.config.adapter_path}, "
                f"output_dir={self.config.output_dir}"
            )
            self.checks_passed += 1
            return True
        except OSError as e:
            logger.error(f"  ❌ Directories: cannot create — {e}")
            self.checks_failed += 1
            return False


# ─────────────────────────────────────────────────────────────────────────────
# Memory Estimator
# ─────────────────────────────────────────────────────────────────────────────

def estimate_memory_usage(config: TrainingConfig) -> dict[str, float]:
    """Estimate peak memory usage for the training configuration.

    This provides a rough pre-training estimate to help users understand
    whether their configuration will fit in memory. Actual usage may vary
    depending on sequence lengths, model architecture, and MLX internals.

    Args:
        config: Training configuration.

    Returns:
        Dict mapping component names to estimated memory in GB.
    """
    # Model size estimation (4-bit quantized 8B parameters)
    # 8B params × 4 bits / 8 bits per byte = ~4 GB
    model_gb = 4.5

    # LoRA adapters: rank × hidden_dim × 2 (up + down) × num_layers × 4 bytes
    # DeepSeek-8B hidden_dim ≈ 4096
    # 8 × 4096 × 2 × 8 layers × 2 (for QKV projections) × 4 bytes ≈ 4 MB
    lora_gb = 0.1

    # Optimizer states (AdamW): 2 states per LoRA parameter
    optimizer_gb = lora_gb * 2

    # Gradients for LoRA layers
    gradients_gb = 0.5

    # Activations: depends on batch_size and seq_length
    # Rough estimate: batch × seq × hidden × layers × bytes_per_activation
    activations_gb = (
        config.batch_size
        * config.max_seq_length
        * 4096  # hidden_dim
        * config.lora_layers
        * 2  # bytes (fp16)
        / (1024 ** 3)
    )
    # Add overhead multiplier
    activations_gb = max(activations_gb * 4, 0.5)

    # Gradient checkpointing reduces activation memory by ~60%
    if config.grad_checkpoint:
        activations_gb *= 0.4

    # Tokenizer + overhead
    overhead_gb = 0.5

    # OS + System
    system_gb = 2.0

    estimates = {
        "Quantized Model (4-bit)": model_gb,
        "LoRA Adapters": lora_gb,
        "Optimizer States": optimizer_gb,
        "Gradients": gradients_gb,
        "Activations": round(activations_gb, 2),
        "Tokenizer + Overhead": overhead_gb,
        "OS + System": system_gb,
    }

    return estimates


def print_memory_estimate(config: TrainingConfig) -> None:
    """Print a formatted memory usage estimate table."""
    estimates = estimate_memory_usage(config)
    total = sum(estimates.values())
    available = psutil.virtual_memory().total / (1024 ** 3)
    margin = available - total

    logger.info("┌─────────────────────────────────────────────┐")
    logger.info("│  ESTIMATED MEMORY USAGE                     │")
    logger.info("├─────────────────────────────┬───────────────┤")

    for component, gb in estimates.items():
        logger.info(f"│  {component:<27} │  {gb:>6.2f} GB     │")

    logger.info("├─────────────────────────────┼───────────────┤")
    logger.info(f"│  {'TOTAL ESTIMATED':<27} │  {total:>6.2f} GB     │")
    logger.info(f"│  {'SYSTEM TOTAL':<27} │  {available:>6.2f} GB     │")
    logger.info(f"│  {'SAFETY MARGIN':<27} │  {margin:>6.2f} GB     │")
    logger.info("└─────────────────────────────┴───────────────┘")

    if margin < 2.0:
        logger.warning(
            "⚠ Low safety margin! Consider reducing batch_size, "
            "lora_layers, or max_seq_length."
        )


# ─────────────────────────────────────────────────────────────────────────────
# YAML Config Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_lora_config(
    config: TrainingConfig,
    config_path: str = "lora_config.yaml",
) -> str:
    """Generate a YAML configuration file for mlx_lm.lora.

    The generated config contains all hyperparameters needed by mlx_lm.lora
    and can be passed via the --config flag.

    Args:
        config: Training configuration dataclass.
        config_path: Path to write the YAML file.

    Returns:
        Path to the generated config file.
    """
    lora_config = {
        # Model
        "model": config.model,
        "trust_remote_code": config.trust_remote_code,

        # LoRA
        "fine_tune_type": config.fine_tune_type,
        "num_layers": config.lora_layers,
        "lora_parameters": {
            "rank": config.lora_rank,
            "scale": config.lora_scale,
            "dropout": config.lora_dropout,
        },

        # Training
        "train": True,
        "iters": config.num_iters,
        "batch_size": config.batch_size,
        "grad_accumulation_steps": config.grad_accumulation_steps,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer,
        "seed": config.seed,

        # LR schedule: None means use constant learning_rate (safe default)
        # To use cosine decay, set lr_schedule to:
        #   {"name": "cosine_decay", "arguments": [1e-5, 600], "warmup": 50}
        "lr_schedule": config.lr_schedule,

        # Sequence
        "max_seq_length": config.max_seq_length,

        # Regularization
        "grad_checkpoint": config.grad_checkpoint,
        "mask_prompt": config.mask_prompt,

        # Data
        "data": config.data_dir,

        # Output
        "adapter_path": config.adapter_path,
        "save_every": config.save_every,
        "steps_per_eval": config.steps_per_eval,
        "steps_per_report": config.steps_per_report,
    }

    with open(config_path, "w") as f:
        yaml.dump(lora_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated LoRA config: {config_path}")
    return config_path


# ─────────────────────────────────────────────────────────────────────────────
# Training Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class TrainingOrchestrator:
    """Orchestrates the complete LoRA fine-tuning pipeline.

    Pipeline stages:
      1. Pre-flight validation
      2. Memory estimation
      3. Config generation
      4. Training execution (via mlx_lm.lora)
      5. Evaluation (optional)
      6. Adapter fusion (optional)
      7. Test generation (optional)
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the training orchestrator.

        Args:
            config: TrainingConfig with all hyperparameters.
        """
        self.config = config
        self.config_path = "lora_config.yaml"

    def run(
        self,
        skip_preflight: bool = False,
        skip_fusion: bool = False,
        skip_test_generation: bool = False,
    ) -> None:
        """Execute the complete training pipeline.

        Args:
            skip_preflight: Skip pre-flight checks (for debugging).
            skip_fusion: Skip post-training adapter fusion.
            skip_test_generation: Skip test generation after training.
        """
        self._print_banner()

        # Stage 1: Pre-flight checks
        if not skip_preflight:
            checker = PreFlightChecker(self.config)
            if not checker.run_all_checks():
                logger.error(
                    "Pre-flight checks failed. Fix the issues above and retry."
                )
                sys.exit(1)

        # Stage 2: Memory estimation
        print_memory_estimate(self.config)

        # Stage 3: Generate config
        self.config_path = generate_lora_config(self.config)

        # Stage 4: Training
        self._run_training()

        # Stage 5: Evaluation
        self._run_evaluation()

        # Stage 6: Fusion (optional)
        if not skip_fusion:
            self._run_fusion()

        # Stage 7: Test generation (optional)
        if not skip_test_generation:
            self._run_test_generation()

        self._print_completion()

    def _print_banner(self) -> None:
        """Print the training banner."""
        logger.info("╔" + "═" * 58 + "╗")
        logger.info("║  Neuro-Algorithmic-Data-Engine                          ║")
        logger.info("║  LoRA Fine-Tuning Pipeline                              ║")
        logger.info("╠" + "═" * 58 + "╣")
        logger.info(f"║  Model:   {self.config.model:<47} ║")
        logger.info(f"║  Type:    {self.config.fine_tune_type:<47} ║")
        logger.info(f"║  Iters:   {self.config.num_iters:<47} ║")
        logger.info(f"║  LR:      {self.config.learning_rate:<47} ║")
        logger.info(f"║  Batch:   {self.config.batch_size} (effective: {self.config.batch_size * self.config.grad_accumulation_steps}){'':<32} ║")
        logger.info(f"║  Layers:  {self.config.lora_layers:<47} ║")
        logger.info(f"║  Rank:    {self.config.lora_rank:<47} ║")
        logger.info("╚" + "═" * 58 + "╝")

    def _run_training(self) -> None:
        """Execute mlx_lm.lora training via subprocess.

        Uses the official mlx_lm.lora CLI with the generated YAML config.
        Subprocess execution ensures clean process isolation and enables
        real-time log streaming.
        """
        logger.info("\n" + "=" * 60)
        logger.info("STAGE: TRAINING")
        logger.info("=" * 60)

        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--config", self.config_path,
        ]

        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("-" * 60)

        try:
            process = subprocess.run(
                cmd,
                check=True,
                text=True,
            )
            logger.info("Training completed successfully! ✅")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed with exit code {e.returncode}")
            logger.error(f"stderr: {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(
                "mlx_lm.lora not found. "
                "Install with: pip install 'mlx-lm[train]'"
            )
            sys.exit(1)

    def _run_evaluation(self) -> None:
        """Run evaluation on the test set using the trained adapter."""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE: EVALUATION")
        logger.info("=" * 60)

        test_file = Path(self.config.data_dir) / "test.jsonl"
        if not test_file.exists():
            logger.warning(
                f"Test file {test_file} not found. Skipping evaluation."
            )
            return

        cmd = [
            sys.executable, "-m", "mlx_lm", "lora",
            "--model", self.config.model,
            "--adapter-path", self.config.adapter_path,
            "--data", self.config.data_dir,
            "--test",
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, text=True)
            logger.info("Evaluation completed! ✅")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Evaluation failed: {e}. Continuing...")

    def _run_fusion(self) -> None:
        """Fuse LoRA adapters into the base model.

        Creates a standalone model with the LoRA weights merged into the
        base model weights. This is useful for deployment and inference
        without needing to load adapters separately.
        """
        logger.info("\n" + "=" * 60)
        logger.info("STAGE: ADAPTER FUSION")
        logger.info("=" * 60)

        adapter_weights = Path(self.config.adapter_path) / "adapters.safetensors"
        if not adapter_weights.exists():
            logger.warning(
                f"Adapter weights not found at {adapter_weights}. "
                f"Skipping fusion."
            )
            return

        cmd = [
            sys.executable, "-m", "mlx_lm", "fuse",
            "--model", self.config.model,
            "--adapter-path", self.config.adapter_path,
            "--save-path", self.config.output_dir,
        ]

        logger.info(f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, text=True)
            logger.info(f"Model fused to: {self.config.output_dir} ✅")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Fusion failed: {e}. Continuing...")

    def _run_test_generation(self) -> None:
        """Generate text using the fine-tuned model for manual evaluation."""
        logger.info("\n" + "=" * 60)
        logger.info("STAGE: TEST GENERATION")
        logger.info("=" * 60)

        adapter_weights = Path(self.config.adapter_path) / "adapters.safetensors"
        if not adapter_weights.exists():
            logger.warning("No adapters found. Skipping test generation.")
            return

        cmd = [
            sys.executable, "-m", "mlx_lm", "generate",
            "--model", self.config.model,
            "--adapter-path", self.config.adapter_path,
            "--prompt", self.config.test_prompt,
            "--max-tokens", str(self.config.max_tokens),
        ]

        logger.info(f"Prompt: {self.config.test_prompt}")
        logger.info("-" * 60)

        try:
            subprocess.run(cmd, check=True, text=True)
            logger.info("Test generation completed! ✅")
        except subprocess.CalledProcessError as e:
            logger.warning(f"Generation failed: {e}")

    def _print_completion(self) -> None:
        """Print the pipeline completion summary."""
        logger.info("\n" + "╔" + "═" * 58 + "╗")
        logger.info("║  PIPELINE COMPLETE                                      ║")
        logger.info("╠" + "═" * 58 + "╣")
        logger.info(f"║  Adapters:    {self.config.adapter_path + '/':<43} ║")
        logger.info(f"║  Fused Model: {self.config.output_dir + '/':<43} ║")
        logger.info(f"║  Config:      {self.config_path:<43} ║")
        logger.info("╠" + "═" * 58 + "╣")
        logger.info("║  Next Steps:                                            ║")
        logger.info("║  1. Review training metrics in the logs above           ║")
        logger.info("║  2. Test with: mlx_lm.generate --model fused_model/     ║")
        logger.info("║  3. Compare with base model to measure improvement      ║")
        logger.info("╚" + "═" * 58 + "╝")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point for the training pipeline.

    Usage:
        # Run with default settings (QLoRA on DeepSeek-8B-4bit)
        python train_mlx.py

        # Custom model and iterations
        python train_mlx.py --model mlx-community/Llama-3.2-3B-Instruct-4bit \\
                            --iters 1000 --learning-rate 2e-5

        # Skip fusion and test generation (training only)
        python train_mlx.py --skip-fusion --skip-test

        # Quick test run
        python train_mlx.py --iters 10 --save-every 5 --val-every 5
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Neuro-Algorithmic-Data-Engine: LoRA Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with defaults (optimized for M4 16GB)
  python train_mlx.py

  # Quick test run (10 iterations)  
  python train_mlx.py --iters 10 --save-every 5 --val-every 5

  # Custom dataset and model
  python train_mlx.py --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \\
                      --data-dir my_data --iters 1000

  # Training only (no fusion or generation)
  python train_mlx.py --skip-fusion --skip-test
        """,
    )

    # Model
    parser.add_argument(
        "--model", type=str,
        default=TrainingConfig.model,
        help=f"Model ID (default: {TrainingConfig.model})",
    )

    # LoRA
    parser.add_argument(
        "--lora-layers", type=int,
        default=TrainingConfig.lora_layers,
        help=f"Number of LoRA layers (default: {TrainingConfig.lora_layers})",
    )
    parser.add_argument(
        "--lora-rank", type=int,
        default=TrainingConfig.lora_rank,
        help=f"LoRA rank (default: {TrainingConfig.lora_rank})",
    )

    # Training
    parser.add_argument(
        "--iters", type=int,
        default=TrainingConfig.num_iters,
        help=f"Training iterations (default: {TrainingConfig.num_iters})",
    )
    parser.add_argument(
        "--batch-size", type=int,
        default=TrainingConfig.batch_size,
        help=f"Batch size (default: {TrainingConfig.batch_size})",
    )
    parser.add_argument(
        "--grad-accumulation", type=int,
        default=TrainingConfig.grad_accumulation_steps,
        help=f"Gradient accumulation steps (default: {TrainingConfig.grad_accumulation_steps})",
    )
    parser.add_argument(
        "--learning-rate", type=float,
        default=TrainingConfig.learning_rate,
        help=f"Learning rate (default: {TrainingConfig.learning_rate})",
    )
    parser.add_argument(
        "--max-seq-length", type=int,
        default=TrainingConfig.max_seq_length,
        help=f"Max sequence length (default: {TrainingConfig.max_seq_length})",
    )

    # Data
    parser.add_argument(
        "--data-dir", type=str,
        default=TrainingConfig.data_dir,
        help=f"Data directory (default: {TrainingConfig.data_dir})",
    )

    # Output
    parser.add_argument(
        "--adapter-path", type=str,
        default=TrainingConfig.adapter_path,
        help=f"Adapter output path (default: {TrainingConfig.adapter_path})",
    )
    parser.add_argument(
        "--save-every", type=int,
        default=TrainingConfig.save_every,
        help=f"Save every N iters (default: {TrainingConfig.save_every})",
    )
    parser.add_argument(
        "--steps-per-eval", type=int,
        default=TrainingConfig.steps_per_eval,
        help=f"Validate every N iters (default: {TrainingConfig.steps_per_eval})",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-preflight", action="store_true",
        help="Skip pre-flight checks",
    )
    parser.add_argument(
        "--skip-fusion", action="store_true",
        help="Skip post-training adapter fusion",
    )
    parser.add_argument(
        "--skip-test", action="store_true",
        help="Skip test generation",
    )

    # Generation
    parser.add_argument(
        "--test-prompt", type=str,
        default=TrainingConfig.test_prompt,
        help="Prompt for test generation",
    )

    args = parser.parse_args()

    # Build configuration from CLI arguments
    config = TrainingConfig(
        model=args.model,
        lora_layers=args.lora_layers,
        lora_rank=args.lora_rank,
        num_iters=args.iters,
        batch_size=args.batch_size,
        grad_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        data_dir=args.data_dir,
        adapter_path=args.adapter_path,
        save_every=args.save_every,
        steps_per_eval=args.steps_per_eval,
        test_prompt=args.test_prompt,
    )

    # Run the pipeline
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run(
        skip_preflight=args.skip_preflight,
        skip_fusion=args.skip_fusion,
        skip_test_generation=args.skip_test,
    )


if __name__ == "__main__":
    main()
