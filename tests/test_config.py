"""Smoke config: defaults y validación que lista TODAS las violaciones."""
import unittest

from scribeforge import config


class TestConfig(unittest.TestCase):
    def test_defaults_pass_validation(self):
        self.assertEqual(config.validate(config.DEFAULTS), [])

    def test_all_violations_listed_together(self):
        bad = {
            "model": {"size": "gigante", "device": "tpu"},
            "summary": {"mode": "churro", "context_chars": 10, "chunk_overlap": 5, "timeout_s": -1},
            "llm": {"endpoints": [{"base_url": "localhost-no-url"}]},
        }
        errs = config.validate(bad)
        self.assertGreaterEqual(len(errs), 6)
        self.assertTrue(any("model.size" in e for e in errs))
        self.assertTrue(any("model.device" in e for e in errs))
        self.assertTrue(any("summary.mode" in e for e in errs))
        self.assertTrue(any("summary.context_chars" in e for e in errs))
        self.assertTrue(any("summary.chunk_overlap" in e for e in errs))
        self.assertTrue(any("llm.endpoints" in e for e in errs))

    def test_partial_config_merges_with_defaults(self):
        cfg = {**config.DEFAULTS, "model": {**config.DEFAULTS["model"], "size": "large-v3"}}
        self.assertEqual(cfg["model"]["size"], "large-v3")
        self.assertEqual(config.validate(cfg), [])


if __name__ == "__main__":
    unittest.main()
