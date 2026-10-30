"""The standalone explainer must remain offline and in sync with its source."""
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_algorithm_overview", ROOT / "tools/render_algorithm_overview.py"
)
renderer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(renderer)


class AlgorithmOverviewDocumentTest(unittest.TestCase):
    def test_generated_document_matches_source(self):
        self.assertEqual(renderer.OUTPUT.read_text(encoding="utf-8"), renderer.render())

    def test_light_standalone_without_remote_dependencies(self):
        document = renderer.render()
        self.assertIn('<html lang="zh-CN" data-theme="light">', document)
        self.assertIn('name="color-scheme" content="light"', document)
        self.assertNotIn("<iframe", document)
        self.assertNotRegex(document, r'<(?:script|link)[^>]+(?:src|href)="https?://')

    def test_complete_paths_and_comparison_are_preserved(self):
        fragment = renderer.SOURCE.read_text(encoding="utf-8")
        for mode in ("o2", "anchor", "regression", "flow", "comparison"):
            self.assertIn('data-mode="' + mode + '"', fragment)
        self.assertIn('id="complete-chapters"', fragment)
        self.assertIn("renderCompleteContent();", fragment)
        self.assertIn('id="chunk-comparison"', fragment)
        self.assertIn("idMap", fragment)
        self.assertIn("beforeprint", fragment)


if __name__ == "__main__":
    unittest.main()
