"""Base prompt class for FinQA."""

from abc import ABC, abstractmethod
from typing import Optional


# DSL operations and description for program synthesis
DSL_DESCRIPTION = """
Write a program to compute the answer using the following DSL operations.

Operations:
- add(a, b): Add two numbers. Example: add(10, 5) = 15
- subtract(a, b): Subtract b from a. Example: subtract(10, 5) = 5
- multiply(a, b): Multiply two numbers. Example: multiply(10, 5) = 50
- divide(a, b): Divide a by b. Example: divide(10, 5) = 2
- exp(a, b): Raise a to the power of b. Example: exp(2, 3) = 8
- greater(a, b): Return "yes" if a > b, else "no". Example: greater(10, 5) = "yes"
- table_sum(row): Sum all values in a table row
- table_average(row): Average of all values in a table row
- table_max(row): Maximum value in a table row
- table_min(row): Minimum value in a table row

Rules:
1. Use #N to refer to the result of step N (0-indexed). Example: #0 refers to the first operation's result.
2. Use const_X for constants: const_100 = 100, const_1000 = 1000, const_1000000 = 1000000.
3. Percentages like "23.6%" should be written as 23.6% in the program.
4. Chain operations sequentially. Each operation can reference previous results with #N.

Example programs:
- "What is 100 divided by 50, then multiplied by 2?" → divide(100, 50), multiply(#0, 2)
- "Is revenue (500) greater than cost (300)?" → greater(500, 300)
- "What is the change from 100 to 150 as a percentage?" → subtract(150, 100), divide(#0, 100), multiply(#1, const_100)
- "What is the sum of 1000 and 2500?" → add(1000, 2500)
- "What is net income if revenue is 5000 and expenses are 3200?" → subtract(5000, 3200)
- "What is the profit margin if profit is 800 and revenue is 4000?" → divide(800, 4000), multiply(#0, const_100)
- "What is 15% of 2000?" → multiply(2000, 15%), divide(#0, const_100)
- "What is the year-over-year growth from 1200 to 1500?" → subtract(1500, 1200), divide(#0, 1200), multiply(#1, const_100)
- "What is the total in millions if the value is 5.2 million?" → multiply(5.2, const_1000000)
""".strip()


class BasePrompt(ABC):
    """Abstract base class for prompt templates."""

    # System prompts for different modes
    SYSTEM_PROMPT_ANSWER = (
        "You are a financial expert. Given the context and question, "
        "provide the numerical answer. Be precise and give only the final number."
    )

    SYSTEM_PROMPT_PROGRAM = (
        "You are a financial expert. Given the context and question, "
        "write a program using the provided DSL to compute the answer. "
        "Output only the program, nothing else."
    )

    def __init__(
        self,
        include_system: bool = True,
        output_program: bool = False,
    ):
        """
        Initialize prompt template.

        Args:
            include_system: Whether to include system prompt
            output_program: If True, prompt for program output; else direct answer
        """
        self.include_system = include_system
        self.output_program = output_program

    @abstractmethod
    def format(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> str | list[dict]:
        """
        Format the prompt.

        Args:
            question: The question to answer
            context: Financial document context
            **kwargs: Additional arguments

        Returns:
            Formatted prompt (string or chat format)
        """
        pass

    def _get_system_prompt(self) -> str:
        """Get appropriate system prompt based on output mode."""
        if self.output_program:
            return self.SYSTEM_PROMPT_PROGRAM
        return self.SYSTEM_PROMPT_ANSWER

    def _to_chat_format(self, user_content: str) -> list[dict]:
        """Convert to chat message format."""
        messages = []
        if self.include_system:
            messages.append({"role": "system", "content": self._get_system_prompt()})
        messages.append({"role": "user", "content": user_content})
        return messages
