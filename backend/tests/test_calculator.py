import pytest

from app.services.tools.calculator import calculator_tool, evaluate


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("2 + 40", 42),
        ("(1250 - 1000) / 1000 * 100", 25.0),
        ("2 ** 10", 1024),
        ("2 ^ 10", 1024),  # models often write ^ for powers
        ("-3 + 5", 2),
        ("sqrt(16)", 4.0),
        ("round(2.567, 2)", 2.57),
        ("10 % 3", 1),
    ],
)
def test_valid_expressions(expression, expected):
    assert evaluate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",  # the reason eval() is forbidden
        "open('/etc/passwd').read()",
        "x + 1",  # names
        "[1, 2, 3]",
        "'a' * 3",  # strings
        "10 ** 10 ** 10",  # would freeze the server
        "1 / 0",
        "2 +",
        "9" * 300,  # too long
    ],
)
def test_dangerous_or_invalid_expressions_are_rejected(expression):
    with pytest.raises(ValueError):
        evaluate(expression)


def test_tool_output_is_readable_for_the_model():
    output = calculator_tool.run({"expression": "(1250 - 1000) / 1000 * 100"})
    assert output.content == "(1250 - 1000) / 1000 * 100 = 25"


def test_tool_schema_is_in_openai_format():
    schema = calculator_tool.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculator"
    assert schema["function"]["parameters"]["required"] == ["expression"]
