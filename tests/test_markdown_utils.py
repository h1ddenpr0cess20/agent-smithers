from agent_smithers.markdown_utils import render_markdown


def test_render_markdown_uses_matrix_safe_fenced_code_html():
    html = render_markdown("**User**:\n```python\nprint(1)\n```")
    assert html is not None
    assert "<pre><code" in html
    assert 'class="language-python"' in html
    assert "codehilite" not in html
    assert "<span" not in html
