from omniresearch.agents.protocol import parse_action, parse_json_object


def test_tool_action():
    a = parse_action('{"thought": "x", "action": "tool", "tool": "python", "input": {"code": "print(1)"}}')
    assert a.kind == "tool" and a.tool == "python" and a.input == {"code": "print(1)"}


def test_fenced_json_with_prose():
    a = parse_action('Sure.\n```json\n{"action": "final", "content": "Answer **here**"}\n```\nThanks')
    assert a.kind == "final" and a.content == "Answer **here**"


def test_plain_text_is_final():
    a = parse_action("The answer is 42.")
    assert a.kind == "final" and a.content == "The answer is 42."


def test_trailing_comma_tolerated():
    a = parse_action('{"action": "tool", "tool": "knowledge_search", "input": {"query": "soil carbon",},}')
    assert a.kind == "tool" and a.input["query"] == "soil carbon"


def test_nested_braces_in_strings():
    a = parse_action('{"action": "tool", "tool": "python", "input": {"code": "d = {\\"a\\": 1}; print(d)"}}')
    assert a.kind == "tool" and 'd = {"a": 1}' in a.input["code"]


def test_alternate_keys():
    a = parse_action('{"tool_name": "wikipedia", "arguments": {"topic": "Permafrost"}}')
    assert a.kind == "tool" and a.tool == "wikipedia" and a.input == {"topic": "Permafrost"}


def test_parse_json_object():
    d = parse_json_object('Plan:\n{"domains": ["ecology"], "subquestions": [{"question": "q"}]}')
    assert d["domains"] == ["ecology"]
