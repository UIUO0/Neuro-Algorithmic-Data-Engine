"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Neuro-Algorithmic-Data-Engine — Inference Script                          ║
║                                                                            ║
║  Interactive Text-to-SQL Demo                                              ║
║  Translates natural language questions into SQL queries using the           ║
║  fine-tuned DeepSeek-8B model on Apple Silicon via MLX.                    ║
║                                                                            ║
║  Author: Neuro-Algorithmic-Data-Engine Team                                ║
║  License: MIT                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage Modes:
─────────────
  1. Interactive mode (REPL):  python3 inference.py
  2. Single query:             python3 inference.py --question "..." --schema "..."

The script automatically detects whether to use the fused model (standalone)
or the base model + LoRA adapters, preferring the fused model for faster
inference.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional

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
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FUSED_MODEL_PATH = "fused_model"
DEFAULT_BASE_MODEL = "mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit"
DEFAULT_ADAPTER_PATH = "adapters"
DEFAULT_MAX_TOKENS = 200

SYSTEM_PROMPT = (
    "You are a SQL expert. Given a user question and the relevant table "
    "schemas, generate the correct SQL query to answer the question."
)

# ANSI color codes for terminal output
COLORS = {
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "CYAN": "\033[96m",
    "MAGENTA": "\033[95m",
    "RED": "\033[91m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "RESET": "\033[0m",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model Loader
# ─────────────────────────────────────────────────────────────────────────────

class TextToSQLEngine:
    """Interactive Text-to-SQL inference engine.

    Loads the fine-tuned model (fused or base+adapter) and provides
    methods for generating SQL from natural language questions.

    Model Resolution Order:
        1. Fused model (preferred — no adapter overhead)
        2. Base model + LoRA adapters
        3. Base model only (no fine-tuning)

    Attributes:
        model: The loaded MLX language model.
        tokenizer: The associated tokenizer.
        model_path: Path to the model being used.
    """

    def __init__(
        self,
        fused_model_path: str = DEFAULT_FUSED_MODEL_PATH,
        base_model: str = DEFAULT_BASE_MODEL,
        adapter_path: str = DEFAULT_ADAPTER_PATH,
    ):
        """Initialize the inference engine.

        Args:
            fused_model_path: Path to the fused (merged) model directory.
            base_model: HuggingFace model ID for the base model.
            adapter_path: Path to LoRA adapter weights.
        """
        self.model = None
        self.tokenizer = None
        self.model_path = None
        self.adapter_path = None
        self._generate_fn = None

        self._load_model(fused_model_path, base_model, adapter_path)

    def _load_model(
        self,
        fused_model_path: str,
        base_model: str,
        adapter_path: str,
    ) -> None:
        """Load the best available model configuration.

        Priority: fused_model > base+adapters > base only.

        Complexity:
            Time:  O(P) where P = model parameters (one-time load)
            Space: O(P) — model weights in memory (~4.5 GB for 4-bit 8B)
        """
        try:
            from mlx_lm import load, generate
        except ImportError:
            logger.error(
                "mlx-lm is not installed. Run: pip install 'mlx-lm[train]'"
            )
            sys.exit(1)

        self._generate_fn = generate

        fused_path = Path(fused_model_path)
        adapter_file = Path(adapter_path) / "adapters.safetensors"

        # Priority 1: Fused model (best — no adapter overhead)
        if fused_path.exists() and (fused_path / "config.json").exists():
            logger.info(f"Loading fused model from: {fused_model_path}")
            self.model, self.tokenizer = load(fused_model_path)
            self.model_path = fused_model_path
            logger.info("✅ Fused model loaded successfully")
            return

        # Priority 2: Base model + LoRA adapters
        if adapter_file.exists():
            logger.info(f"Loading base model: {base_model}")
            logger.info(f"Applying LoRA adapters from: {adapter_path}")
            self.model, self.tokenizer = load(
                base_model,
                adapter_path=adapter_path,
            )
            self.model_path = base_model
            self.adapter_path = adapter_path
            logger.info("✅ Base model + adapters loaded successfully")
            return

        # Priority 3: Base model only (no fine-tuning)
        logger.warning(
            "No fused model or adapters found. "
            "Loading base model without fine-tuning."
        )
        self.model, self.tokenizer = load(base_model)
        self.model_path = base_model
        logger.info("⚠️  Base model loaded (no fine-tuning applied)")

    def build_prompt(self, question: str, schema: str = "") -> str:
        """Build a structured prompt for Text-to-SQL generation.

        Formats the input into the same structure used during training:
            ### Table Schema:
            CREATE TABLE ...

            ### Question:
            What is the average salary ...?

        Args:
            question: Natural language question.
            schema: SQL table schema (CREATE TABLE statements).

        Returns:
            Formatted prompt string.
        """
        parts = []

        if schema.strip():
            parts.append(f"### Table Schema:\n{schema.strip()}")

        parts.append(f"### Question:\n{question.strip()}")

        return "\n\n".join(parts)

    def generate_sql(
        self,
        question: str,
        schema: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> str:
        """Generate a SQL query from a natural language question.

        Args:
            question: Natural language question about the data.
            schema: SQL table schema (CREATE TABLE statements).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy/deterministic).

        Returns:
            Generated SQL query string.
        """
        prompt = self.build_prompt(question, schema)

        # Build chat-format messages (same structure as training)
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback for tokenizers without chat template
            formatted_prompt = (
                f"System: {SYSTEM_PROMPT}\n\n"
                f"User: {prompt}\n\n"
                f"Assistant: "
            )

        # Generate SQL
        response = self._generate_fn(
            self.model,
            self.tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
        )

        # Clean up the response
        sql = response.strip()

        # Remove common generation artifacts
        for stop_token in ["<|end|>", "<|im_end|>", "</s>", "<eos>"]:
            if stop_token in sql:
                sql = sql[:sql.index(stop_token)].strip()

        return sql


# ─────────────────────────────────────────────────────────────────────────────
# Interactive REPL
# ─────────────────────────────────────────────────────────────────────────────

def print_banner():
    """Print the interactive mode welcome banner."""
    C = COLORS
    print(f"""
{C['BOLD']}{C['CYAN']}╔══════════════════════════════════════════════════════════════╗
║  🧠 Neuro-Algorithmic-Data-Engine                            ║
║  Interactive Text-to-SQL Generator                           ║
╠══════════════════════════════════════════════════════════════╣
║  Type your question in natural language.                     ║
║  Commands:                                                   ║
║    /schema  — Set or change the table schema                 ║
║    /clear   — Clear the current schema                       ║
║    /example — Show an example query                          ║
║    /quit    — Exit the program                               ║
╚══════════════════════════════════════════════════════════════╝{C['RESET']}
""")


def print_example():
    """Print an example interaction."""
    C = COLORS
    print(f"""
{C['BOLD']}Example:{C['RESET']}

{C['DIM']}Schema:{C['RESET']}
  CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary DECIMAL);
  CREATE TABLE departments (id INT, name VARCHAR, budget DECIMAL);

{C['DIM']}Question:{C['RESET']}
  What is the average salary for each department?

{C['GREEN']}Generated SQL:{C['RESET']}
  SELECT department, AVG(salary) FROM employees GROUP BY department;
""")


def run_interactive(engine: TextToSQLEngine, max_tokens: int):
    """Run the interactive REPL loop.

    Provides a terminal-based interface for entering table schemas
    and natural language questions, with colored output.

    Args:
        engine: Loaded TextToSQLEngine instance.
        max_tokens: Maximum tokens to generate per query.
    """
    C = COLORS
    print_banner()

    current_schema = ""

    while True:
        try:
            # Get user input
            print(f"{C['BOLD']}{C['BLUE']}┌─ Question (or /command):{C['RESET']}")
            user_input = input(f"{C['BLUE']}└─▶ {C['RESET']}").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ("/quit", "/exit", "/q"):
                print(f"\n{C['DIM']}Goodbye! 👋{C['RESET']}")
                break

            elif user_input.lower() == "/schema":
                print(f"\n{C['YELLOW']}Enter table schema (CREATE TABLE statements).")
                print(f"Press Enter twice to finish:{C['RESET']}")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        if lines and lines[-1].strip() == "":
                            break
                        lines.append(line)
                    else:
                        lines.append(line)
                current_schema = "\n".join(lines).strip()
                print(f"{C['GREEN']}✅ Schema set! ({len(current_schema)} chars){C['RESET']}\n")
                continue

            elif user_input.lower() == "/clear":
                current_schema = ""
                print(f"{C['GREEN']}✅ Schema cleared.{C['RESET']}\n")
                continue

            elif user_input.lower() == "/example":
                print_example()
                continue

            elif user_input.startswith("/"):
                print(f"{C['RED']}Unknown command: {user_input}{C['RESET']}")
                print(f"{C['DIM']}Available: /schema, /clear, /example, /quit{C['RESET']}\n")
                continue

            # Show current schema context
            if current_schema:
                schema_preview = current_schema[:80] + "..." if len(current_schema) > 80 else current_schema
                print(f"{C['DIM']}  Using schema: {schema_preview}{C['RESET']}")

            # Generate SQL
            print(f"{C['DIM']}  Generating...{C['RESET']}", end="", flush=True)

            sql = engine.generate_sql(
                question=user_input,
                schema=current_schema,
                max_tokens=max_tokens,
            )

            # Display result
            print(f"\r{' ' * 40}\r", end="")  # Clear "Generating..." line
            print(f"\n{C['BOLD']}{C['GREEN']}┌─ Generated SQL:{C['RESET']}")
            print(f"{C['GREEN']}│{C['RESET']}")

            for line in sql.split("\n"):
                print(f"{C['GREEN']}│  {C['BOLD']}{line}{C['RESET']}")

            print(f"{C['GREEN']}│{C['RESET']}")
            print(f"{C['GREEN']}└{'─' * 60}{C['RESET']}\n")

        except KeyboardInterrupt:
            print(f"\n\n{C['DIM']}Interrupted. Type /quit to exit.{C['RESET']}\n")
        except EOFError:
            print(f"\n{C['DIM']}Goodbye! 👋{C['RESET']}")
            break


# ─────────────────────────────────────────────────────────────────────────────
# Single Query Mode
# ─────────────────────────────────────────────────────────────────────────────

def run_single_query(
    engine: TextToSQLEngine,
    question: str,
    schema: str,
    max_tokens: int,
    output_json: bool = False,
):
    """Run a single Text-to-SQL query and print the result.

    Args:
        engine: Loaded TextToSQLEngine instance.
        question: Natural language question.
        schema: SQL table schema.
        max_tokens: Maximum tokens to generate.
        output_json: If True, output as JSON for scripting.
    """
    sql = engine.generate_sql(
        question=question,
        schema=schema,
        max_tokens=max_tokens,
    )

    if output_json:
        result = {
            "question": question,
            "schema": schema,
            "generated_sql": sql,
            "model": engine.model_path,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        C = COLORS
        print(f"\n{C['BOLD']}{C['CYAN']}Question:{C['RESET']} {question}")
        if schema:
            print(f"{C['DIM']}Schema:{C['RESET']}   {schema[:100]}...")
        print(f"\n{C['BOLD']}{C['GREEN']}SQL:{C['RESET']}      {sql}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point for the inference script.

    Usage Examples:
        # Interactive mode (REPL)
        python3 inference.py

        # Single query
        python3 inference.py \\
          --question "What is the average salary by department?" \\
          --schema "CREATE TABLE employees (id INT, name VARCHAR, department VARCHAR, salary DECIMAL);"

        # JSON output (for scripting/pipelines)
        python3 inference.py \\
          --question "How many orders per customer?" \\
          --schema "CREATE TABLE orders (id INT, customer_id INT, amount DECIMAL);" \\
          --json
    """
    parser = argparse.ArgumentParser(
        description="Neuro-Algorithmic-Data-Engine: Text-to-SQL Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python3 inference.py

  # Single query with schema
  python3 inference.py \\
    --question "What is the average salary by department?" \\
    --schema "CREATE TABLE employees (id INT, name VARCHAR, dept VARCHAR, salary DECIMAL);"

  # JSON output for pipelines
  python3 inference.py --question "Count all users" --json

  # Use a specific model path
  python3 inference.py --model fused_model/
        """,
    )

    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Natural language question (runs single-query mode)",
    )
    parser.add_argument(
        "--schema", "-s",
        type=str,
        default="",
        help="SQL table schema (CREATE TABLE statements)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_FUSED_MODEL_PATH,
        help=f"Model path — fused model dir or HF ID (default: {DEFAULT_FUSED_MODEL_PATH})",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=DEFAULT_ADAPTER_PATH,
        help=f"LoRA adapter path (default: {DEFAULT_ADAPTER_PATH})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum tokens to generate (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Sampling temperature; 0.0 = deterministic (default: 0.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output results as JSON (useful for scripting)",
    )

    args = parser.parse_args()

    # Load model
    engine = TextToSQLEngine(
        fused_model_path=args.model,
        base_model=DEFAULT_BASE_MODEL,
        adapter_path=args.adapter_path,
    )

    # Route to appropriate mode
    if args.question:
        # Single query mode
        run_single_query(
            engine=engine,
            question=args.question,
            schema=args.schema,
            max_tokens=args.max_tokens,
            output_json=args.output_json,
        )
    else:
        # Interactive REPL mode
        run_interactive(engine, max_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
