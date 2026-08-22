"""MCP tools offered to the Tool-box challenge agent."""

from __future__ import annotations

import ast
from typing import Literal

from fastmcp import FastMCP

from .shapes import classify_shape_image


mcp = FastMCP("Tool-box Nursery")


@mcp.tool(
    name="get_name",
    description="Return the assistant's name as a short valid string.",
)
def get_name() -> str:
    return "Jinx Club"


@mcp.tool(
    name="calculate",
    description="Evaluate an integer expression using BODMAS: brackets, then *, /, +, and -.",
)
def calculate(expression: str) -> int | float:
    """Evaluate one expression without using Python's unsafe ``eval``."""

    if len(expression) > 500:
        raise ValueError("expression is too long")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError("expression is not valid arithmetic") from error

    result = _evaluate_expression(tree.body)
    return int(result) if isinstance(result, float) and result.is_integer() else result


def _evaluate_expression(node: ast.expr) -> int | float:
    if isinstance(node, ast.Constant):
        if type(node.value) is not int or not -100 <= node.value <= 100:
            raise ValueError("every operand must be an integer from -100 to 100")
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_expression(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    if not isinstance(node, ast.BinOp):
        raise ValueError("only brackets and +, -, *, / are supported")

    left = _evaluate_expression(node.left)
    right = _evaluate_expression(node.right)
    if isinstance(node.op, ast.Add):
        return left + right
    if isinstance(node.op, ast.Sub):
        return left - right
    if isinstance(node.op, ast.Mult):
        return left * right
    if isinstance(node.op, ast.Div):
        if right == 0:
            raise ValueError("division by zero is undefined")
        return left / right

    raise ValueError("only brackets and +, -, *, / are supported")


@mcp.tool(
    name="classify_shape",
    description="Classify one base64 PNG as rectangle, triangle, or circle.",
)
def classify_shape(image_base64: str) -> Literal["rectangle", "triangle", "circle"]:
    return classify_shape_image(image_base64)  # type: ignore[return-value]
