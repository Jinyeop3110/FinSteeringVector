"""Chain-of-thought prompt template for FinQA."""

import re
from typing import Optional, List, Dict, Any
from .base import BasePrompt, DSL_DESCRIPTION


def parse_program(program: str) -> List[tuple]:
    """Parse program into list of (operation, [args])"""
    steps = []
    parts = re.split(r',\s*(?=[a-z_]+\()', program)

    for part in parts:
        part = part.strip()
        match = re.match(r'([a-z_]+)\(([^)]+)\)', part)
        if match:
            op = match.group(1)
            args = [a.strip() for a in match.group(2).split(',')]
            steps.append((op, args))
    return steps


def execute_program_with_trace(program: str) -> tuple:
    """Execute program and return (trace_lines, final_result)"""
    steps = parse_program(program)
    results = {}
    trace_lines = []

    for i, (op, args) in enumerate(steps):
        # Resolve arguments
        resolved = []
        for arg in args:
            if arg.startswith('#'):
                ref_idx = int(arg[1:])
                resolved.append(results.get(ref_idx, 0))
            elif arg.startswith('const_'):
                const_val = int(arg.replace('const_', ''))
                resolved.append(const_val)
            elif arg.endswith('%'):
                resolved.append(float(arg[:-1]) / 100)
            else:
                try:
                    resolved.append(float(arg))
                except:
                    resolved.append(arg)

        # Execute operation
        result = None
        if op == 'add' and len(resolved) == 2:
            result = resolved[0] + resolved[1]
        elif op == 'subtract' and len(resolved) == 2:
            result = resolved[0] - resolved[1]
        elif op == 'multiply' and len(resolved) == 2:
            result = resolved[0] * resolved[1]
        elif op == 'divide' and len(resolved) == 2 and resolved[1] != 0:
            result = resolved[0] / resolved[1]
        elif op == 'greater' and len(resolved) == 2:
            result = 'yes' if resolved[0] > resolved[1] else 'no'
        elif op == 'exp' and len(resolved) == 2:
            result = resolved[0] ** resolved[1]

        if result is not None:
            results[i] = result
            args_str = ', '.join(str(a) for a in args)
            if isinstance(result, float):
                trace_lines.append(f"Step {i+1}: {op}({args_str}) = {result:.5g}")
            else:
                trace_lines.append(f"Step {i+1}: {op}({args_str}) = {result}")

    final_result = results.get(max(results.keys())) if results else None
    return trace_lines, final_result


def generate_reasoning(program: str, gold_inds: Dict[str, str] = None) -> str:
    """Generate full reasoning trace from program and gold_inds"""
    lines = []

    # Add evidence from gold_inds
    if gold_inds:
        lines.append("Evidence from context:")
        for key, value in list(gold_inds.items())[:3]:  # Limit to 3
            # Clean up the evidence text
            value_clean = value[:150] + "..." if len(value) > 150 else value
            lines.append(f"  - {value_clean}")
        lines.append("")

    # Add execution trace
    trace_lines, _ = execute_program_with_trace(program)
    if trace_lines:
        lines.append("Calculation:")
        lines.extend(trace_lines)

    return '\n'.join(lines)


class ChainOfThoughtPrompt(BasePrompt):
    """Chain-of-thought prompt that encourages step-by-step reasoning."""

    SYSTEM_PROMPT_ANSWER = (
        "You are a financial expert. Given the context and question, "
        "think step by step to solve the problem. First identify the relevant evidence, "
        "then show your calculations, and provide the final numerical answer."
    )

    SYSTEM_PROMPT_PROGRAM = (
        "You are a financial expert. Given the context and question, "
        "think step by step: identify the evidence, plan the calculations, "
        "then write the program using the DSL. Output only the program on the last line."
    )

    TEMPLATE_ANSWER = """Context:
{context}

Question: {question}

Let's solve this step by step:
1. First, identify the relevant numbers from the context.
2. Determine what calculation is needed.
3. Perform the calculation.
4. State the final answer.

Solution:"""

    TEMPLATE_PROGRAM = """{dsl_description}

Context:
{context}

Question: {question}

Let's think about what operations we need:
1. Identify the relevant values from the context.
2. Determine the sequence of operations.
3. Write the program.

Program:"""

    EXAMPLE_TEMPLATE_ANSWER = """Context:
{context}

Question: {question}

Solution:
{reasoning}

Answer: {answer}

---
"""

    EXAMPLE_TEMPLATE_PROGRAM = """Context:
{context}

Question: {question}

Reasoning: {reasoning}

Program: {program}

---
"""

    def __init__(
        self,
        n_shots: int = 2,
        include_system: bool = True,
        output_program: bool = False,
        max_context_len: int = 1000,
    ):
        """
        Initialize chain-of-thought prompt.

        Args:
            n_shots: Number of examples to include
            include_system: Whether to include system prompt
            output_program: If True, prompt for program output; else direct answer
            max_context_len: Maximum context length per example
        """
        super().__init__(include_system, output_program)
        self.n_shots = n_shots
        self.max_context_len = max_context_len

    def _get_system_prompt(self) -> str:
        """Get appropriate system prompt for CoT."""
        if self.output_program:
            return self.SYSTEM_PROMPT_PROGRAM
        return self.SYSTEM_PROMPT_ANSWER

    def _truncate_context(self, context: str) -> str:
        """Truncate context to max length."""
        if len(context) <= self.max_context_len:
            return context
        return context[:self.max_context_len] + "..."

    def format_example(self, example: dict) -> str:
        """Format a single CoT example with auto-generated reasoning."""
        context = self._truncate_context(example.get("context", ""))

        # Auto-generate reasoning from program and gold_inds
        program = example.get("program", "")
        gold_inds = example.get("gold_inds", {})

        if program:
            reasoning = generate_reasoning(program, gold_inds)
        else:
            reasoning = "Extracting values and computing the result."

        if self.output_program:
            return self.EXAMPLE_TEMPLATE_PROGRAM.format(
                context=context,
                question=example["question"],
                reasoning=reasoning,
                program=program,
            )
        else:
            return self.EXAMPLE_TEMPLATE_ANSWER.format(
                context=context,
                question=example["question"],
                reasoning=reasoning,
                answer=example["answer"],
            )

    def format(
        self,
        question: str,
        context: str,
        icl_examples: Optional[list[dict]] = None,
        **kwargs,
    ) -> str | list[dict]:
        """
        Format chain-of-thought prompt.

        Args:
            question: The question to answer
            context: Financial document context
            icl_examples: Optional ICL examples

        Returns:
            Formatted prompt in chat format
        """
        parts = []

        # Add ICL examples if provided
        if icl_examples:
            if self.output_program:
                parts.append(DSL_DESCRIPTION)
                parts.append("")
            parts.append("Here are some examples of step-by-step solutions:\n")
            for ex in icl_examples[:self.n_shots]:
                parts.append(self.format_example(ex))
            parts.append("Now solve the following:\n")

        # Add the query
        if self.output_program:
            if not icl_examples:
                parts.append(
                    self.TEMPLATE_PROGRAM.format(
                        dsl_description=DSL_DESCRIPTION,
                        context=context,
                        question=question,
                    )
                )
            else:
                parts.append(f"""Context:
{context}

Question: {question}

Program:""")
        else:
            if not icl_examples:
                parts.append(
                    self.TEMPLATE_ANSWER.format(
                        context=context,
                        question=question,
                    )
                )
            else:
                parts.append(f"""Context:
{context}

Question: {question}

Solution:""")

        user_content = "\n".join(parts)
        return self._to_chat_format(user_content)
