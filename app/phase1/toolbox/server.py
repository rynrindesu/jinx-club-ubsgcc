"""MCP tools offered to the Tool-box challenge agent."""

from __future__ import annotations

from typing import Annotated, Literal

from fastmcp import FastMCP
from pydantic import Field

from .shapes import classify_shape_image


mcp = FastMCP("Tool-box Nursery")
Operand = Annotated[int, Field(ge=-100, le=100)]


@mcp.tool(
    name="get_name",
    description="Return the assistant's name as a short valid string.",
)
def get_name() -> str:
    return "Jinx Club"


@mcp.tool(
    name="calculate",
    description="Calculate two integers using +, -, *, or /.",
)
def calculate(
    left: Operand,
    operator: Literal["+", "-", "*", "/"],
    right: Operand,
) -> int | float:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if right == 0:
        raise ValueError("division by zero is undefined")

    result = left / right
    return int(result) if result.is_integer() else result


@mcp.tool(
    name="classify_shape",
    description="Classify one base64 PNG as rectangle, triangle, or circle.",
)
def classify_shape(image_base64: str) -> Literal["rectangle", "triangle", "circle"]:
    return classify_shape_image(image_base64)  # type: ignore[return-value]
