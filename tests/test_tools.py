from pathlib import Path

from omniresearch.tools.knowledge import KnowledgeBase, knowledge_tools
from omniresearch.tools.memory import Memory, memory_tools
from omniresearch.tools.python_exec import python_tool
from omniresearch.tools.registry import ToolRegistry
from omniresearch.tools.textutil import chunk_text, fts_query, html_to_text


def test_knowledge_index_and_search(tmp_path: Path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "permafrost.md").write_text("# Permafrost\nThawing permafrost releases methane and carbon dioxide, "
                                        "amplifying warming in the Arctic.\n" * 3)
    (docs / "soil.txt").write_text("Soil organic carbon improves water retention and crop yield.")
    (docs / "page.html").write_text("<html><body><h1>Coral</h1><p>Bleaching occurs above 1 degree C anomaly.</p></body></html>")
    kb = KnowledgeBase(tmp_path / "kb.db")
    indexed, skipped = kb.index_path(docs)
    assert indexed == 3
    hits = kb.search("permafrost methane")
    assert hits and "permafrost" in hits[0]["path"]
    assert kb.search("coral bleaching")[0]["path"].endswith("page.html")
    # re-index is a no-op for unchanged files
    assert kb.index_path(docs) == (0, 3)
    tools = ToolRegistry(knowledge_tools(kb))
    res = tools.run("knowledge_search", query="crop yield soil")
    assert not res.error and "soil" in res.output.lower() and res.sources
    res = tools.run("knowledge_read", path=hits[0]["path"])
    assert "methane" in res.output
    assert tools.run("knowledge_read", path="missing").error


def test_add_text_and_stats(tmp_path: Path):
    kb = KnowledgeBase(tmp_path / "kb.db")
    kb.add_text("Wikipedia: Glacier", "A glacier is a persistent body of dense ice.", path="https://x/glacier")
    assert kb.stats()["documents"] == 1
    assert kb.search("dense ice")[0]["path"] == "https://x/glacier"


def test_memory(tmp_path: Path):
    mem = Memory(tmp_path / "m.db")
    tools = ToolRegistry(memory_tools(mem, "s1"))
    tools.run("memory_save", topic="Heat islands", content="Urban heat islands raise night temps by 2-5 C", tags="urban")
    out = tools.run("memory_recall", query="urban heat")
    assert "Heat islands" in out.output
    assert mem.count() == 1


def test_python_tool(tmp_path: Path):
    tool = python_tool(tmp_path, timeout=10)
    assert tool.run(code="print(round(math.sqrt(2), 3))").output.strip() == "1.414"
    assert tool.run(code="```python\nprint(2+2)\n```").output.strip() == "4"
    assert tool.run(code="raise ValueError('x')").error
    assert tool.run(code="import time; time.sleep(30)").error


def test_registry_errors():
    reg = ToolRegistry()
    assert reg.run("nope").error
    tool = python_tool(Path("."))
    assert tool.run().error  # missing required


def test_text_helpers():
    assert html_to_text("<p>Hi <b>there</b></p><script>x</script>") == "Hi there"
    assert fts_query("coral-reef bleaching!") == '"coral-reef" OR "bleaching"'
    chunks = chunk_text("word " * 2000, size=500, overlap=50)
    assert len(chunks) > 10 and all(len(c) <= 520 for c in chunks)
