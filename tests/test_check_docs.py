"""Offline documentation checks, including broken-link failure cases."""
import tempfile
import unittest
from pathlib import Path

from tools.check_docs import DEFAULT_PATHS, ROOT, check_files, heading_anchors


class DocumentationLinksTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def write(self, name, text):
        file = self.root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
        return file

    def test_unicode_duplicate_headings_and_explicit_anchors(self):
        anchors = heading_anchors(
            '# 数据与 `TCP`：配置\n## 重复\n## 重复\n## 重复-1\n'
            '<a id="legacy-entry"></a>\n```markdown\n## Not a heading\n```\n'
        )
        self.assertEqual(anchors, {"数据与-tcp配置", "重复", "重复-1", "重复-1-1", "legacy-entry"})

    def test_relative_encoded_targets_and_images(self):
        self.write("参考 空间.md", "# 运行说明\n")
        self.write("assets/example.svg", "<svg/>")
        source = self.write("guides/start.md", '[配置](../参考%20空间.md#运行说明)\n![图](../assets/example.svg)\n')
        self.assertEqual(check_files([source]), (2, []))

    def test_missing_file_and_anchor_are_failures(self):
        self.write("target.md", "# Present\n")
        source = self.write("source.md", "[missing](absent.md)\n[wrong](target.md#absent)\n")
        count, issues = check_files([source])
        self.assertEqual(count, 2)
        self.assertEqual(len(issues), 2)
        self.assertIn("missing file", issues[0])
        self.assertIn(":2:", issues[1])
        self.assertIn("missing heading", issues[1])

    def test_code_examples_and_external_urls_are_excluded(self):
        source = self.write("source.md", "```markdown\n[example](absent.md)\n```\n"
                            "`[example](absent.md)`\n[web](https://example.invalid/page)\n")
        self.assertEqual(check_files([source]), (0, []))

    def test_reference_definitions_and_html_targets(self):
        source = self.write("source.md", '[ref]: missing.md\n<img src="absent.svg">\n')
        count, issues = check_files([source])
        self.assertEqual(count, 2)
        self.assertEqual(len(issues), 2)

    def test_repository_documentation_has_no_broken_local_links(self):
        files = []
        for name in DEFAULT_PATHS:
            path = ROOT / name
            files.extend(path.rglob("*.md") if path.is_dir() else [path])
        count, issues = check_files(sorted(files))
        self.assertGreater(count, 0)
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
