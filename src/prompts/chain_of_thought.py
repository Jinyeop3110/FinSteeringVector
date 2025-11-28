"""Chain-of-thought prompt template for FinQA."""

import re
from typing import Optional
from .base import BasePrompt


def parse_program(program: str) -> list[tuple[str, list[str]]]:
    """
    Parse a FinQA program into list of (operation, arguments).

    Example: "subtract(959.2, 991.1), divide(#0, 991.1)"
    Returns: [("subtract", ["959.2", "991.1"]), ("divide", ["#0", "991.1"])]
    """
    steps = []
    # Match operation(arg1, arg2, ...)
    pattern = r'(\w+)\(([^)]+)\)'
    matches = re.findall(pattern, program)
    for op, args_str in matches:
        args = [a.strip() for a in args_str.split(',')]
        steps.append((op, args))
    return steps


def extract_numbers_from_gold_inds(gold_inds: dict, col_name: str) -> list[float]:
    """
    Extract numeric values from gold_inds that match a column/row name.

    Examples:
    - "table_1": "the structured commercial loan vehicles of 2003 is $ 5.3 ; ... of 2002 is $ 7.2 ;"
      -> [5.3, 7.2]
    - "table_2": "the liabilities of credit card is 1824 ; ... of automobile is 250 ;"
      -> [1824, 250]
    - "table_5": "the settlements of 2013 is -603 ( 603 ) ; ... of 2012 is -67 ( 67 ) ;"
      -> [-603, -67] (signed value, not the parenthetical duplicate)
    - "table_3": "the expected volatility of 2005 is 84% ( 84 % ) ;"
      -> [84] (percentage value without the duplicate)

    Strategy: For each semicolon-separated entry, look for "is VALUE" pattern.
    Handle: "is -603 ( 603 )" -> -603, "is 84% ( 84 % )" -> 84
    """
    numbers = []
    col_name_lower = col_name.lower().replace("_", " ")

    for key, value in gold_inds.items():
        if not isinstance(value, str):
            continue
        value_lower = value.lower()

        # Check if this gold_ind is about the column we're looking for
        if col_name_lower in value_lower or col_name_lower.replace(" ", "") in value_lower.replace(" ", ""):
            # Split by semicolons to get individual value entries
            entries = value.split(';')
            for entry in entries:
                entry = entry.strip()
                if not entry:
                    continue

                # Strategy: Look for "is VALUE" pattern where VALUE is immediately after "is"
                # This handles: "is -603 ( 603 )", "is 84% ( 84 % )", "is $ 5.3"
                # The value right after "is" is the canonical one (with sign/symbol)

                # Pattern: "is" followed by optional whitespace, then value
                # Value can be: -$123.45, -123.45, 123.45%, $123, etc.
                is_pattern = r'is\s+(-?\$?\s*[\d,]+\.?\d*)\s*%?'
                is_match = re.search(is_pattern, entry)

                if is_match:
                    num_str = is_match.group(1)
                    try:
                        # Clean the number string
                        clean_str = num_str.replace('$', '').replace(',', '').replace(' ', '').strip()
                        if not clean_str or clean_str == '-':
                            continue
                        num = float(clean_str)
                        # Filter out likely year values (1900-2100) unless they have decimals
                        if 1900 <= abs(num) <= 2100 and num == int(num):
                            continue
                        numbers.append(num)
                    except ValueError:
                        pass
                else:
                    # Fallback: find the last valid number in the entry
                    num_pattern = r'(-?\$?\s*[\d,]+\.?\d*)'
                    all_nums = re.findall(num_pattern, entry)

                    for num_str in reversed(all_nums):
                        try:
                            clean_str = num_str.replace('$', '').replace(',', '').replace(' ', '').strip()
                            if not clean_str or clean_str == '-':
                                continue
                            num = float(clean_str)
                            if 1900 <= abs(num) <= 2100 and num == int(num):
                                continue
                            numbers.append(num)
                            break
                        except ValueError:
                            continue

    return numbers


def execute_steps(steps: list[tuple[str, list[str]]], gold_inds: dict = None, answer: str = None) -> list[dict]:
    """
    Execute program steps and return intermediate results with explanations.

    Args:
        steps: Parsed program steps
        gold_inds: Evidence dictionary for extracting table values
        answer: Ground truth answer (used to infer table operation results)
    """
    results = {}
    explanations = []
    gold_inds = gold_inds or {}

    for i, (op, args) in enumerate(steps):
        # Handle table operations - extract actual values from gold_inds
        if op in ("table_sum", "table_average", "table_max", "table_min"):
            col_name = args[0] if args else "values"
            numbers = extract_numbers_from_gold_inds(gold_inds, col_name)

            result = None
            explanation = ""

            if numbers:
                if op == "table_sum":
                    result = sum(numbers)
                    nums_str = " + ".join(str(int(n) if n == int(n) else n) for n in numbers)
                    explanation = f"{nums_str} = {int(result) if result == int(result) else round(result, 4)}"
                elif op == "table_average":
                    result = sum(numbers) / len(numbers)
                    nums_str = " + ".join(str(int(n) if n == int(n) else n) for n in numbers)
                    explanation = f"({nums_str}) / {len(numbers)} = {round(result, 4)}"
                elif op == "table_max":
                    result = max(numbers)
                    nums_str = ", ".join(str(int(n) if n == int(n) else n) for n in numbers)
                    explanation = f"max({nums_str}) = {int(result) if result == int(result) else round(result, 4)}"
                elif op == "table_min":
                    result = min(numbers)
                    nums_str = ", ".join(str(int(n) if n == int(n) else n) for n in numbers)
                    explanation = f"min({nums_str}) = {int(result) if result == int(result) else round(result, 4)}"

                if result is not None:
                    results[i] = result
            else:
                # Fallback: just describe the operation
                explanation = f"Calculate {op.replace('table_', '')} of '{col_name}' from the table"

            explanations.append({
                "op": op,
                "args": args,
                "result": result,
                "explanation": explanation,
                "step_idx": i,
                "is_table_op": True,
                "extracted_numbers": numbers,
            })
            continue

        # Resolve arguments (replace #n references with previous results)
        resolved_args = []
        arg_strs = []
        for arg in args:
            if arg.startswith('#'):
                ref_idx = int(arg[1:])
                if ref_idx in results:
                    resolved_args.append(results[ref_idx])
                    arg_strs.append(str(round(results[ref_idx], 4)))
                else:
                    resolved_args.append(None)
                    arg_strs.append(arg)
            elif arg.startswith('const_'):
                # Handle constants like const_1000, const_100, const_1000000
                const_val = arg.replace('const_', '')
                try:
                    val = float(const_val)
                    resolved_args.append(val)
                    arg_strs.append(const_val)
                except:
                    resolved_args.append(None)
                    arg_strs.append(arg)
            else:
                # Try to parse as number
                try:
                    # Check if it's a percentage
                    if '%' in arg:
                        val = float(arg.replace('%', '').replace(',', '')) / 100.0
                        resolved_args.append(val)
                        arg_strs.append(arg)  # Keep display as "23.6%"
                    else:
                        val = float(arg.replace(',', ''))
                        resolved_args.append(val)
                        arg_strs.append(arg)
                except:
                    resolved_args.append(None)
                    arg_strs.append(arg)

        # Skip if we couldn't resolve all args
        if None in resolved_args:
            continue

        # Execute operation
        result = None
        explanation = ""

        if op == "add" and len(resolved_args) == 2:
            result = resolved_args[0] + resolved_args[1]
            explanation = f"{arg_strs[0]} + {arg_strs[1]} = {round(result, 4)}"
        elif op == "subtract" and len(resolved_args) == 2:
            result = resolved_args[0] - resolved_args[1]
            explanation = f"{arg_strs[0]} - {arg_strs[1]} = {round(result, 4)}"
        elif op == "multiply" and len(resolved_args) == 2:
            result = resolved_args[0] * resolved_args[1]
            explanation = f"{arg_strs[0]} × {arg_strs[1]} = {round(result, 4)}"
        elif op == "divide" and len(resolved_args) == 2:
            if resolved_args[1] != 0:
                result = resolved_args[0] / resolved_args[1]
                explanation = f"{arg_strs[0]} ÷ {arg_strs[1]} = {round(result, 4)}"
        elif op == "greater" and len(resolved_args) == 2:
            result = 1.0 if resolved_args[0] > resolved_args[1] else 0.0
            comparison = ">" if resolved_args[0] > resolved_args[1] else "≤"
            yes_no = "yes" if resolved_args[0] > resolved_args[1] else "no"
            explanation = f"{arg_strs[0]} {comparison} {arg_strs[1]}, so the answer is {yes_no}"
        elif op == "exp" and len(resolved_args) == 2:
            result = resolved_args[0] ** resolved_args[1]
            explanation = f"{arg_strs[0]} ^ {arg_strs[1]} = {round(result, 4)}"

        if result is not None:
            results[i] = result
            explanations.append({
                "op": op,
                "args": arg_strs,
                "result": result,
                "explanation": explanation,
                "step_idx": i,
            })

    return explanations


def infer_question_intent(question: str, program: str = "") -> dict:
    """
    Infer the intent and key aspects of a financial question.
    Uses both question text and program to determine intent.
    Returns dict with: intent_type, target_metric, comparison_type, time_period
    """
    q_lower = question.lower()

    # Use program to help disambiguate intent
    has_divide = "divide" in program if program else False

    # Determine intent type - be more specific about percentage vs absolute change
    if any(word in q_lower for word in ["percentage change", "percent change", "% change", "growth rate"]):
        intent_type = "percentage_change"
    elif "change" in q_lower and has_divide:
        # "change" + division in program = percentage change
        intent_type = "percentage_change"
    elif "change" in q_lower and not has_divide:
        # "change" without division = absolute difference
        intent_type = "difference"
    elif any(word in q_lower for word in ["ratio", "proportion", "percentage of", "% of", "fraction"]):
        intent_type = "ratio"
    elif any(word in q_lower for word in ["difference", "how much more", "how much less", "increase", "decrease"]):
        intent_type = "difference"
    elif any(word in q_lower for word in ["total", "sum", "combined", "aggregate"]):
        intent_type = "sum"
    elif any(word in q_lower for word in ["average", "mean"]):
        intent_type = "average"
    elif any(word in q_lower for word in ["greater", "larger", "higher", "more than", "exceed"]):
        intent_type = "comparison"
    else:
        intent_type = "calculation"

    # Determine time comparison
    time_comparison = None
    if "year over year" in q_lower or "yoy" in q_lower:
        time_comparison = "year_over_year"
    elif any(p in q_lower for p in ["from 2018 to 2019", "from 2019 to 2020", "2018 to 2019", "2019 to 2020"]):
        time_comparison = "year_to_year"
    elif "quarter" in q_lower:
        time_comparison = "quarter"

    return {
        "intent_type": intent_type,
        "time_comparison": time_comparison,
    }


def generate_semantic_explanation(op: str, args: list[str], step_idx: int,
                                   prev_explanations: list[dict], question: str) -> str:
    """
    Generate a semantic explanation for why an operation is performed.
    """
    q_lower = question.lower()

    # Check if this uses a previous result
    uses_prev_result = step_idx > 0 and len(prev_explanations) > 0

    if op == "subtract":
        if uses_prev_result:
            return "Using this result, we subtract to find the net difference."
        elif "change" in q_lower or "difference" in q_lower:
            return f"To find the change, we subtract the earlier value ({args[1]}) from the later value ({args[0]})."
        else:
            return f"We subtract {args[1]} from {args[0]} to find the difference."

    elif op == "divide":
        if uses_prev_result:
            if "percent" in q_lower or "%" in q_lower or "change" in q_lower:
                return f"To express this as a percentage, we divide by the base value ({args[1]})."
            else:
                return f"We divide by {args[1]} to find the ratio."
        elif "ratio" in q_lower or "proportion" in q_lower:
            return f"To find the ratio, we divide {args[0]} by {args[1]}."
        elif "percent" in q_lower or "%" in q_lower:
            return f"To calculate the percentage, we divide {args[0]} by {args[1]}."
        else:
            return f"We divide {args[0]} by {args[1]}."

    elif op == "add":
        if "total" in q_lower or "sum" in q_lower:
            return f"To find the total, we add {args[0]} and {args[1]} together."
        else:
            return f"We add {args[0]} and {args[1]}."

    elif op == "multiply":
        if uses_prev_result:
            return f"We multiply by {args[1]} to scale the result."
        elif "100" in args[1]:
            return "We multiply by 100 to convert to a percentage."
        else:
            return f"We multiply {args[0]} by {args[1]}."

    elif op == "greater":
        return f"We compare {args[0]} and {args[1]} to determine which is larger."

    elif op == "exp":
        return f"We raise {args[0]} to the power of {args[1]}."

    # Table operations
    elif op == "table_sum":
        col_name = args[0] if args else "the column"
        if "total" in q_lower:
            return f"To find the total, we sum all values in '{col_name}' from the table."
        return f"We sum all the values in '{col_name}' from the table."

    elif op == "table_average":
        col_name = args[0] if args else "the column"
        return f"To find the average, we calculate the mean of all values in '{col_name}' from the table."

    elif op == "table_max":
        col_name = args[0] if args else "the column"
        if "greatest" in q_lower or "highest" in q_lower or "maximum" in q_lower:
            return f"To find the greatest value, we identify the maximum in '{col_name}' from the table."
        return f"We find the maximum value in '{col_name}' from the table."

    elif op == "table_min":
        col_name = args[0] if args else "the column"
        if "smallest" in q_lower or "lowest" in q_lower or "minimum" in q_lower:
            return f"To find the smallest value, we identify the minimum in '{col_name}' from the table."
        return f"We find the minimum value in '{col_name}' from the table."

    return f"We perform {op} on the values."


class ChainOfThoughtPrompt(BasePrompt):
    """Chain-of-thought prompt that encourages step-by-step reasoning."""

    SYSTEM_PROMPT = (
        "You are a financial expert. Given the context and question, "
        "think step by step to solve the problem. Show your reasoning briefly, "
        "then provide the final answer on a new line starting with 'Answer:'."
    )

    TEMPLATE = """Context:
{context}

Question: {question}

Think step by step and show your reasoning. Then provide your final answer on a new line starting with "Answer:".
For numerical questions, give only the number.
For yes/no questions, answer with "yes" or "no".

Reasoning:"""

    TEMPLATE_NO_CONTEXT = """Context: ***

Question: {question}

Think step by step and show your reasoning. Then provide your final answer on a new line starting with "Answer:".
For numerical questions, give only the number.
For yes/no questions, answer with "yes" or "no".

Reasoning:"""

    # Template with context (default)
    EXAMPLE_TEMPLATE = """Context:
{context}

Question: {question}

Reasoning: {reasoning}

Answer: {answer}

---
"""

    # Template without context (question + reasoning + answer only)
    EXAMPLE_TEMPLATE_NO_CONTEXT = """Context: ***

Question: {question}

Reasoning: {reasoning}

Answer: {answer}

---
"""

    # Template with table only (question + table + reasoning + answer)
    EXAMPLE_TEMPLATE_TABLE_ONLY = """Table:
{table}

Question: {question}

Reasoning: {reasoning}

Answer: {answer}

---
"""

    def __init__(
        self,
        n_shots: int = 2,
        include_system: bool = True,
        max_context_len: int = 20000,
        include_context_in_examples: bool = True,
        table_only_in_examples: bool = False,
        no_context_in_query: bool = False,
    ):
        """
        Initialize chain-of-thought prompt.

        Args:
            n_shots: Number of examples to include
            include_system: Whether to include system prompt
            max_context_len: Maximum context length per example
            include_context_in_examples: Whether to include context in ICL examples
            table_only_in_examples: Whether to include only table (no text) in ICL examples
            no_context_in_query: Whether to exclude context from the main query
        """
        self.n_shots = n_shots
        self.include_system = include_system
        self.max_context_len = max_context_len
        self.include_context_in_examples = include_context_in_examples
        self.table_only_in_examples = table_only_in_examples
        self.no_context_in_query = no_context_in_query

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _to_chat_format(self, user_content: str) -> list[dict]:
        """Convert to chat message format."""
        messages = []
        if self.include_system:
            messages.append({"role": "system", "content": self._get_system_prompt()})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _truncate_context(self, context: str) -> str:
        """Truncate context to max length."""
        if len(context) <= self.max_context_len:
            return context
        return context[:self.max_context_len] + "..."

    def _generate_reasoning(self, example: dict) -> str:
        """Generate step-by-step reasoning trace from the example with semantic explanations."""
        gold_inds = example.get("gold_inds", {})
        program = example.get("program", "")
        answer = example.get("answer", "")
        question = example.get("question", "")

        lines = []

        # Step 0: Interpret what the question is asking (use program to help disambiguate)
        intent = infer_question_intent(question, program)
        intent_type = intent["intent_type"]

        if intent_type == "percentage_change":
            lines.append("The question asks for a percentage change, so we need to find the difference and divide by the base value.")
        elif intent_type == "ratio":
            lines.append("The question asks for a ratio or proportion, so we need to divide one value by another.")
        elif intent_type == "difference":
            lines.append("The question asks for the difference between values.")
        elif intent_type == "sum":
            lines.append("The question asks for a total, so we need to add the values together.")
        elif intent_type == "comparison":
            lines.append("The question asks us to compare values to determine which is larger.")

        # Step 1: Extract key evidence from gold_inds with semantic context
        if gold_inds:
            evidence_items = list(gold_inds.items())[:3]
            evidence_parts = []
            for key, value in evidence_items:
                value_clean = value[:150] + "..." if len(value) > 150 else value
                evidence_parts.append(value_clean)

            if evidence_parts:
                evidence_text = ", and ".join(evidence_parts)
                lines.append(f"From the context, we identify the relevant values: {evidence_text}.")

        # Step 2: Parse and execute the program with semantic explanations
        if program:
            steps = parse_program(program)
            explanations = execute_steps(steps, gold_inds=gold_inds, answer=answer)

            if explanations:
                lines.append("")

                for i, exp in enumerate(explanations):
                    # Get semantic explanation for why we do this step
                    semantic = generate_semantic_explanation(
                        exp['op'],
                        exp['args'],
                        exp['step_idx'],
                        explanations[:i],
                        question
                    )
                    # Show semantic explanation + calculation (if available)
                    lines.append(semantic)
                    # Show calculation if we have one (table ops now can have calculations too)
                    if exp.get('explanation') and exp.get('result') is not None:
                        lines.append(exp['explanation'])
                    elif exp.get('is_table_op') and not exp.get('extracted_numbers'):
                        # Table op without extracted numbers - skip calculation line
                        pass
                    elif not exp.get('is_table_op'):
                        lines.append(exp['explanation'])

                # Add final answer interpretation
                final_result = explanations[-1].get('result')
                if final_result is None:
                    pass  # Table operations - no numeric result to interpret
                elif explanations[-1]['op'] == 'greater':
                    pass  # Already includes yes/no in explanation
                elif abs(final_result) < 1 and final_result != 0:
                    pct = final_result * 100
                    if intent_type == "percentage_change":
                        sign = "increase" if pct > 0 else "decrease"
                        lines.append(f"Therefore, this represents a {abs(round(pct, 2))}% {sign}.")
                    else:
                        lines.append(f"This equals {round(pct, 2)}% or {round(final_result, 4)} as a decimal.")
            else:
                # Fallback if we couldn't parse the program
                if "subtract" in program and "divide" in program:
                    lines.append("We calculate the change and divide to find the percentage change.")
                elif "subtract" in program:
                    lines.append("We calculate the difference between these values.")
                elif "divide" in program:
                    lines.append("We divide to find the ratio/percentage.")
                elif "add" in program:
                    lines.append("We add these values together.")
                elif "multiply" in program:
                    lines.append("We multiply these values.")
                elif "greater" in program:
                    lines.append("We compare these values to determine which is larger.")
                elif "table_sum" in program:
                    lines.append("We sum the values in the table.")
                elif "table_average" in program:
                    lines.append("We calculate the average of the values.")
                elif "table_max" in program:
                    lines.append("We find the maximum value.")
                elif "table_min" in program:
                    lines.append("We find the minimum value.")
                elif "exp" in program:
                    lines.append("We raise to the power (exponentiation).")

        return "\n".join(lines) if lines else "From the context, we find the relevant information and compute the answer."

    def _extract_table_from_context(self, context: str) -> str:
        """Extract table section from context string."""
        if "Table:" in context:
            # Find the Table: section
            start = context.find("Table:")
            # Find where table ends (next double newline or end)
            end = context.find("\n\n", start + 6)
            if end == -1:
                end = len(context)
            return context[start + 6:end].strip()
        return ""

    def format_example(self, example: dict) -> str:
        """Format a single CoT example."""
        reasoning = self._generate_reasoning(example)

        if self.table_only_in_examples:
            # Table only mode - include table but not text passages
            # Try to get table from metadata first
            table = ""
            metadata = example.get("metadata", {})
            if metadata and "table" in metadata:
                # Convert table list to text format
                table_data = metadata["table"]
                if table_data and isinstance(table_data, list):
                    # Format: header row, then data rows
                    rows = []
                    if len(table_data) > 0:
                        header = table_data[0]
                        for row in table_data[1:]:
                            row_parts = []
                            for h, v in zip(header, row):
                                if v and str(v).strip():
                                    row_parts.append(f"{h}: {v}")
                            if row_parts:
                                rows.append(" | ".join(row_parts))
                        table = " ; ".join(rows)

            # Fallback: extract from context string
            if not table:
                table = self._extract_table_from_context(example.get("context", ""))

            if table:
                return self.EXAMPLE_TEMPLATE_TABLE_ONLY.format(
                    table=table,
                    question=example["question"],
                    reasoning=reasoning,
                    answer=example["answer"],
                )
            else:
                # No table available, fall back to no-context
                return self.EXAMPLE_TEMPLATE_NO_CONTEXT.format(
                    question=example["question"],
                    reasoning=reasoning,
                    answer=example["answer"],
                )
        elif self.include_context_in_examples:
            context = self._truncate_context(example.get("context", ""))
            return self.EXAMPLE_TEMPLATE.format(
                context=context,
                question=example["question"],
                reasoning=reasoning,
                answer=example["answer"],
            )
        else:
            # No context - just question, reasoning, answer
            return self.EXAMPLE_TEMPLATE_NO_CONTEXT.format(
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
    ) -> list[dict]:
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
            parts.append("Here are some examples of step-by-step solutions:\n")
            for ex in icl_examples[:self.n_shots]:
                parts.append(self.format_example(ex))
            parts.append("Now solve the following:\n")

        # Add the query
        if self.no_context_in_query:
            parts.append(self.TEMPLATE_NO_CONTEXT.format(
                question=question,
            ))
        else:
            parts.append(self.TEMPLATE.format(
                context=context,
                question=question,
            ))

        user_content = "\n".join(parts)
        return self._to_chat_format(user_content)
