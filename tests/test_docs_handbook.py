"""Content coverage, offline resources and namespaced navigation for the handbook."""
import hashlib
import html
import importlib.util
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from tools import build_docs_handbook as handbook


HAS_MARKDOWN = importlib.util.find_spec("markdown_it") is not None


class Inventory(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.links, self.images, self.documents = [], [], [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img":
            self.images.append(attrs["src"])
        if tag == "article" and attrs.get("class") == "book-document":
            self.documents.append(attrs["id"])


class HandbookDiscoveryTest(unittest.TestCase):
    def test_all_docs_and_both_entrypoints_are_included_once(self):
        paths = [path for _, group in handbook.discover_sources() for path in group]
        expected = set((handbook.ROOT / "docs").rglob("*.md"))
        expected.update(handbook.ROOT / name for name in ("README.md", "finetune/OHT/README.md"))
        self.assertEqual(set(paths), expected)
        self.assertEqual(len(paths), len(expected))

    def test_new_markdown_is_discovered_without_editing_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("README.md", "finetune/OHT/README.md", "docs/new/topic.md"):
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("# Topic\n", encoding="utf-8")
            groups = handbook.discover_sources(root)
            self.assertIn(root / "docs/new/topic.md", groups[-1][1])


@unittest.skipUnless(HAS_MARKDOWN, "Install tools/requirements-docs.txt to test handbook rendering")
class HandbookRenderingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = handbook.build()

    def test_generated_file_is_in_sync(self):
        self.assertEqual(handbook.OUTPUT.read_text(encoding="utf-8"), self.document)

    def test_coverage_unique_ids_and_local_document_links(self):
        parsed = Inventory()
        parsed.feed(self.document)
        expected = {handbook.source_id(path.relative_to(handbook.ROOT))
                    for _, group in handbook.discover_sources() for path in group}
        self.assertEqual(set(parsed.documents), expected)
        self.assertEqual(len(parsed.ids), len(set(parsed.ids)))
        for url in parsed.links:
            if url.startswith("#doc-"):
                self.assertIn(url[1:], parsed.ids, url)

    def test_offline_light_resources_and_full_algorithm(self):
        self.assertIn('<html lang="zh-CN" data-theme="light">', self.document)
        self.assertNotIn("<iframe", self.document)
        self.assertNotRegex(self.document, r'<(?:script|link)[^>]+(?:src|href)="https?://')
        parsed = Inventory()
        parsed.feed(self.document)
        self.assertTrue(parsed.images)
        self.assertTrue(all(src.startswith("data:image/") for src in parsed.images))
        self.assertIn(handbook.ALGORITHM.read_text(encoding="utf-8"), self.document)
        self.assertLess(len(self.document.encode("utf-8")), 2 * 1024 * 1024)

    def test_source_provenance_is_complete_and_correct(self):
        opening = '<script type="application/json" id="book-source-manifest">'
        manifest = json.loads(self.document.split(opening, 1)[1].split("</script>", 1)[0])
        self.assertEqual(len(manifest), sum(len(group) for _, group in handbook.discover_sources()) + 1)
        for item in manifest:
            self.assertEqual(item["sha256"], hashlib.sha256((handbook.ROOT / item["source"]).read_bytes()).hexdigest())

    def test_unicode_links_duplicate_headings_tables_details_and_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, other = root / "source.md", root / "other.md"
            source.write_text('# 数据与 `TCP`：配置\n## 重复\n## 重复\n'
                              '<a id="legacy"></a>\n\n'
                              '[内部](#重复-1) [跨文档](other.md#目标)\n\n'
                              '| 字段 | 值 |\n| --- | --- |\n| a | b |\n\n'
                              '<details>\n<summary>展开</summary>\n\n**完整内容**\n\n</details>\n\n'
                              '```bash\necho "$VALUE" > output.txt\n```\n', encoding="utf-8")
            other.write_text("# 目标\n", encoding="utf-8")
            identities = {source.resolve(): "doc-source", other.resolve(): "doc-other"}
            _, content, _ = handbook.render_markdown(source, identities, root)
            parsed = Inventory()
            parsed.feed(content)
            self.assertIn("doc-source--数据与-tcp配置", parsed.ids)
            self.assertIn("doc-source--重复-1", parsed.ids)
            self.assertIn("doc-source--legacy", parsed.ids)
            self.assertIn("#doc-source--重复-1", parsed.links)
            self.assertIn("#doc-other--目标", parsed.links)
            self.assertIn("<table>", content)
            self.assertIn("<details>", content)
            self.assertIn("<strong>完整内容</strong>", content)
            self.assertIn('echo &quot;$VALUE&quot; &gt; output.txt', content)

    def test_active_html_and_unsafe_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.md"
            source.write_text('<a href="javascript:alert(1)">unsafe</a>', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported source URL"):
                handbook.render_markdown(source, {source.resolve(): "doc-source"}, Path(directory))

    def test_editorial_chapters_keep_commands_while_adapting_navigation_and_tables(self):
        self.assertIn('class="doc-introduction"', self.document)
        self.assertIn('class="doc-source"', self.document)
        self.assertIn('class="long-example"', self.document)
        self.assertIn('class="wide-table"', self.document)
        self.assertIn('data-label="Model"', self.document)
        self.assertIn("Qᶠ′ = Qᶠ + W_o", self.document)

    def test_every_fenced_command_and_source_diagram_is_retained(self):
        from markdown_it import MarkdownIt
        parser = MarkdownIt("commonmark", {"html": True})
        decoded = html.unescape(self.document)
        for _, paths in handbook.discover_sources():
            for path in paths:
                for token in parser.parse(path.read_text(encoding="utf-8-sig")):
                    if token.type == "fence":
                        self.assertTrue(token.content in decoded, "Missing fenced block in " + str(path) + ": " + repr(token.content[:80]))


if __name__ == "__main__":
    unittest.main()
