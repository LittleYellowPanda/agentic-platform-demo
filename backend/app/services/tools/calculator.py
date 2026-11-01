"""Calculator tool. LLMs are unreliable at arithmetic; this tool is exact.

Never `eval()`: the expression is written by the model, i.e. indirectly by whoever
wrote the prompt. eval("__import__('os').system('rm -rf /')") would run that
command. Instead, the expression is parsed into a syntax tree and only a short
list of number operations is allowed.
"""

import ast
import math
import operator

from app.services.tools import Tool, ToolOutput

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {"sqrt": math.sqrt, "log": math.log, "exp": math.exp, "abs": abs, "round": round}

MAX_EXPONENT = 100  # 10 ** 10 ** 10 would freeze the server
MAX_LENGTH = 200


def evaluate(expression: str) -> float | int:
    """Evaluate an arithmetic expression safely. Raises ValueError if not allowed."""
    if len(expression) > MAX_LENGTH:
        raise ValueError(f"expression longer than {MAX_LENGTH} characters")
    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"not a valid expression: {expression!r}") from error
    return _eval(tree.body)


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent larger than {MAX_EXPONENT}")
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as error:
            raise ValueError("division by zero") from error
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_eval(arg) for arg in node.args))
    raise ValueError(f"not allowed: {ast.unparse(node)}")


def _run(arguments: dict) -> ToolOutput:
    expression = str(arguments.get("expression", ""))
    result = evaluate(expression)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return ToolOutput(content=f"{expression} = {result}")


calculator_tool = Tool(
    name="calculator",
    description=(
        "Evaluate an arithmetic expression exactly. Use it for any calculation "
        "(percentages, returns, sums) instead of computing in your head. "
        "Supports + - * / // % ** and sqrt, log, exp, abs, round."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "For example: (1250 - 1000) / 1000 * 100",
            }
        },
        "required": ["expression"],
    },
    run=_run,
)
