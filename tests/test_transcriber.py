"""Smoke Fase 2: post-procesado con segmentos sintéticos (sin modelo, sin red)."""
import unittest

from scribeforge.models import Segment
from scribeforge import transcriber as tr


def seg(s, e, text, nsp=0.0, lp=-0.2):
    return Segment(s, e, text, nsp, lp)


class TestPostProcess(unittest.TestCase):
    def test_filter_hallucinations(self):
        segs = [seg(0, 1, "Hola, bienvenidos."),
                seg(1, 2, "Subtítulos por la comunidad Amara.org", nsp=0.1),
                seg(2, 3, "Y entonces dijo la verdad.", nsp=0.9),
                seg(3, 4, "Y entonces dijo la verdad.", lp=-1.5)]
        out = tr.filter_hallucinations(segs)
        self.assertEqual([s.text for s in out], ["Hola, bienvenidos."])

    def test_dedupe_consecutive(self):
        segs = [seg(0, 1, "funciona"), seg(1, 2, "funciona"), seg(2, 3, "otra cosa")]
        out = tr.dedupe_consecutive(segs)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].end, 2)

    def test_paragraphs_by_pause_and_sentence(self):
        segs = [seg(0, 1, "Primera frase."),
                seg(1.2, 2, "Segunda frase."),
                seg(5.0, 6, "Tercera, tras pausa larga.")]
        out = tr.to_paragraphs(segs)
        self.assertEqual(out.count("\n\n"), 1)
        self.assertIn("Primera frase. Segunda frase.", out)

    def test_paragraphs_capped_by_sentences(self):
        segs = [seg(0 + i * 0.1, 0.5 + i * 0.1, f"Frase número {i} termina.") for i in range(5)]
        out = tr.to_paragraphs(segs)
        self.assertEqual(out.count("\n\n"), 1)  # 5 frases, tope 3 → párrafos 3+2

    def test_clean_fillers(self):
        self.assertEqual(tr.clean_fillers("Eh, esto, el plan funciona"), "el plan funciona")

    def test_srt_format(self):
        srt = tr.to_srt([seg(0, 1.5, "Hola")])
        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nHola\n", srt)

    def test_save_outputs(self):
        import tempfile, json
        from pathlib import Path
        from scribeforge.models import Transcript
        outdir = Path(tempfile.mkdtemp())
        t = Transcript(segments=[seg(0, 1, "Hola mundo.")], language="es", model="base", duration=1.0)
        paths = tr.save_outputs(t, outdir, clean=True)
        self.assertTrue(all(p.exists() for p in paths.values()))
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["language"], "es")
        self.assertIn("Hola mundo.", paths["txt"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
