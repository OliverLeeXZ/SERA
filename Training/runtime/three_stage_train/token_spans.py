from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class SpanParseResult:
    spans: tuple[tuple[int, int], ...]
    attempted_delegation: bool
    valid_ast: bool


def _is_launch_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "launch_subagent"
    return isinstance(func, ast.Attribute) and func.attr == "launch_subagent"


def _line_byte_col_to_char(source_line: str, byte_col: int) -> int:
    encoded = source_line.encode("utf-8")[:byte_col]
    return len(encoded.decode("utf-8", errors="ignore"))


def _node_char_span(source: str, node: ast.AST) -> tuple[int, int]:
    lines = source.splitlines(keepends=True)
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        raise ValueError("AST node has no source location")
    start_line = int(node.lineno) - 1
    end_line = int(node.end_lineno) - 1
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line))
    start = line_starts[start_line] + _line_byte_col_to_char(
        lines[start_line], int(node.col_offset)
    )
    end = line_starts[end_line] + _line_byte_col_to_char(
        lines[end_line], int(node.end_col_offset)
    )
    return start, end


def find_delegation_statement_spans(source: str) -> SpanParseResult:
    attempted = "launch_subagent" in source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return SpanParseResult(
            spans=((0, len(source)),) if attempted else (),
            attempted_delegation=attempted,
            valid_ast=False,
        )

    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent

    statements: set[ast.stmt] = set()
    for node in ast.walk(tree):
        if not _is_launch_call(node):
            continue
        current = node
        while current in parents and not isinstance(current, ast.stmt):
            current = parents[current]
        if isinstance(current, ast.stmt):
            statements.add(current)

    spans = tuple(sorted({_node_char_span(source, statement) for statement in statements}))
    return SpanParseResult(
        spans=spans,
        attempted_delegation=attempted,
        valid_ast=True,
    )


def locate_code_spans_in_completion(
    completion_text: str,
    code: str,
    code_spans: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    start = completion_text.find(code)
    if start < 0:
        stripped = code.strip()
        start = completion_text.find(stripped)
        if start < 0:
            return ()
        leading = len(code) - len(code.lstrip())
        return tuple(
            (start + max(0, begin - leading), start + max(0, end - leading))
            for begin, end in code_spans
        )
    return tuple((start + begin, start + end) for begin, end in code_spans)


def _token_boundary_for_char(
    tokenizer: Any,
    output_tokens: list[int],
    target_char: int,
) -> int:
    low, high = 0, len(output_tokens)
    while low < high:
        middle = (low + high) // 2
        text = tokenizer.decode(output_tokens[:middle], skip_special_tokens=False)
        if len(text) < target_char:
            low = middle + 1
        else:
            high = middle
    return low


def output_token_mask_for_char_spans(
    tokenizer: Any,
    output_tokens: list[int],
    spans: Iterable[tuple[int, int]],
) -> list[int]:
    mask = [0] * len(output_tokens)
    for start_char, end_char in spans:
        if end_char <= start_char:
            continue
        start_token = max(
            0,
            _token_boundary_for_char(tokenizer, output_tokens, start_char + 1) - 1,
        )
        end_token = _token_boundary_for_char(tokenizer, output_tokens, end_char)
        end_token = min(len(output_tokens), max(start_token + 1, end_token))
        for index in range(start_token, end_token):
            mask[index] = 1
    return mask
