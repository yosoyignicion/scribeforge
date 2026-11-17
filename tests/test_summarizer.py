"""Smoke Fase 3: chunker + map-reduce con LLM stub (sin red)."""
import unittest

from scribeforge import summarizer
from scribeforge.llm import Endpoint, LLMClient, LLMError


class FakeLLM(LLMClient):
    def __init__(self, fail_times=0):
        self.calls = 0
        self.fail_times = fail_times
        super().__init__([])

    def complete(self, system, user):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise LLMError("stub caído")
        if "Consolídalos" in user or "Consolídalos" in user or "consol" in user.lower():
            return "# Resumen final\nConsolidado de todo."
        return f"RESUMEN-PARCIAL-{self.calls}"


class TestChunker(unittest.TestCase):
    def test_short_text_single_chunk(self):
        self.assertEqual(summarizer.chunk_by_chars("hola", 100), ["hola"])

    def test_empty_text(self):
        self.assertEqual(summarizer.chunk_by_chars("", 100), [])

    def test_long_text_chunks_capped(self):
        text = "\n".join(f"línea {i} con contenido para test" for i in range(200))
        chunks = summarizer.chunk_by_chars(text, 300, overlap_ratio=0.1)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 300 + 30 for c in chunks))  # margen del solape
        self.assertTrue(text.startswith(chunks[0][:5]))
        tail = text[-20:]
        self.assertTrue(chunks[-1].rstrip().endswith(tail.rstrip()))

    def test_invalid_max(self):
        with self.assertRaises(ValueError):
            summarizer.chunk_by_chars("x", 0)


class TestSummarize(unittest.TestCase):
    def _transcript_text(self, n=50):
        return " ".join(f"Oración número {i} del contenido." for i in range(n))

    def test_single_pass(self):
        llm = FakeLLM()
        from scribeforge.models import Transcript
        t = Transcript()
        out = summarizer.summarize(t, self._transcript_text(), "meeting", llm, context_chars=10_000)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(out, "RESUMEN-PARCIAL-1")

    def test_map_reduce(self):
        llm = FakeLLM()
        from scribeforge.models import Transcript
        t = Transcript()
        out = summarizer.summarize(t, self._transcript_text(400), "meeting", llm, context_chars=300)
        self.assertGreater(llm.calls, 1)
        self.assertIn("Consolidado", out)


class TestLLMRouting(unittest.TestCase):
    def test_fallback_to_second_endpoint(self):
        class DeadFirst(LLMClient):
            pass
        # stub a nivel Endpoint: primer endpoint cae, segundo responde
        from scribeforge import llm as llmmod

        calls = []
        real_chat = llmmod.chat

        def fake_chat(ep, system, user, *, timeout=1.0):
            calls.append(ep.name)
            if ep.name == "dead":
                raise LLMError("caído")
            return "ok"

        llmmod.chat = fake_chat
        try:
            client = LLMClient([Endpoint("dead", "http://x/v1", "m"),
                                Endpoint("alive", "http://y/v1", "m")])
            self.assertEqual(client.complete("s", "u"), "ok")
            self.assertEqual(calls, ["dead", "alive"])
        finally:
            llmmod.chat = real_chat

    def test_all_fail(self):
        from scribeforge import llm as llmmod
        real_chat = llmmod.chat
        llmmod.chat = lambda ep, s, u, *, timeout=1.0: (_ for _ in ()).throw(LLMError("caído"))
        try:
            client = LLMClient([Endpoint("a", "http://x/v1", "m")])
            with self.assertRaises(LLMError):
                client.complete("s", "u")
        finally:
            llmmod.chat = real_chat

    def test_prompts_exist(self):
        for mode in summarizer.MODES:
            self.assertTrue(summarizer.load_prompt(mode).strip())


if __name__ == "__main__":
    unittest.main()
