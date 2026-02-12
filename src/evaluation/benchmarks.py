"""
Benchmark tasks for model evaluation.
Includes standard prompts and evaluation tasks.
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class BenchmarkTask:
    """A single benchmark task."""
    prompt: str
    expected_output: str
    category: str


class BenchmarkSuite:
    """
    Collection of benchmark tasks for evaluating fine-tuned models.
    Includes simple Q&A, reasoning, and instruction-following tasks.
    """

    def __init__(self):
        self.tasks = self._load_default_tasks()

    def _load_default_tasks(self) -> List[BenchmarkTask]:
        """Load default benchmark tasks."""
        tasks = []

        # Simple factual Q&A
        tasks.extend([
            BenchmarkTask(
                prompt="What is the capital of France?",
                expected_output="Paris",
                category="factual_qa"
            ),
            BenchmarkTask(
                prompt="Who wrote 'Romeo and Juliet'?",
                expected_output="Shakespeare",
                category="factual_qa"
            ),
            BenchmarkTask(
                prompt="What is 15 + 27?",
                expected_output="42",
                category="math"
            ),
        ])

        # Instruction following
        tasks.extend([
            BenchmarkTask(
                prompt="List three primary colors.",
                expected_output="red, blue, yellow",
                category="instruction_following"
            ),
            BenchmarkTask(
                prompt="Define 'machine learning' in one sentence.",
                expected_output="learning from data",
                category="instruction_following"
            ),
        ])

        # Simple reasoning
        tasks.extend([
            BenchmarkTask(
                prompt="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                expected_output="5 minutes",
                category="reasoning"
            ),
        ])

        return tasks

    def get_prompts_by_category(self, category: str) -> List[str]:
        """Get all prompts for a specific category."""
        return [task.prompt for task in self.tasks if task.category == category]

    def get_all_prompts(self) -> List[str]:
        """Get all benchmark prompts."""
        return [task.prompt for task in self.tasks]

    def get_task_pairs(self) -> List[Tuple[str, str]]:
        """Get (prompt, expected_output) pairs for all tasks."""
        return [(task.prompt, task.expected_output) for task in self.tasks]

    def get_tasks_by_category(self, category: str) -> List[BenchmarkTask]:
        """Get all tasks for a specific category."""
        return [task for task in self.tasks if task.category == category]

    def add_custom_task(self, prompt: str, expected_output: str, category: str = "custom"):
        """Add a custom benchmark task."""
        self.tasks.append(BenchmarkTask(prompt, expected_output, category))

    def add_custom_tasks(self, tasks: List[BenchmarkTask]):
        """Add multiple custom benchmark tasks."""
        self.tasks.extend(tasks)

    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(set(task.category for task in self.tasks))

    def get_category_counts(self) -> Dict[str, int]:
        """Get count of tasks per category."""
        counts = {}
        for task in self.tasks:
            counts[task.category] = counts.get(task.category, 0) + 1
        return counts


class AlpacaEvalPrompts:
    """
    Sample prompts in Alpaca format for testing instruction-tuned models.
    """

    @staticmethod
    def format_alpaca_prompt(instruction: str, input_text: str = "") -> str:
        """Format a prompt in Alpaca style."""
        if input_text:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""

    @staticmethod
    def get_sample_prompts() -> List[str]:
        """Get sample Alpaca-formatted prompts for testing."""
        instructions = [
            "Explain what machine learning is in simple terms.",
            "Write a haiku about coding.",
            "List the steps to make a peanut butter sandwich.",
            "Describe the water cycle.",
            "What are the benefits of exercise?",
        ]

        return [AlpacaEvalPrompts.format_alpaca_prompt(inst) for inst in instructions]
