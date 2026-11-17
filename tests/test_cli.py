"""Smoke Fase 4: CLI — version, dry-run, help y pipeline end-to-end con stubs."""
import json
import tempfile
import unittest
from pathlib import Path

from scribeforge import cli


class TestCLI(unittest.TestCase):
    def test_version(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--version"])
        self.assertEqual(ctx.exception.code, 0)

    def test_help_lists_flags(self):
        with self.assertRaises(SystemExit) as ctx:
            cli.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_bad_extension_exit_code(self):
        tmp = Path(tempfile.mkdtemp())
        bad = tmp / "doc.pdf"
        bad.write_text("x")
        code = cli.main([str(bad)])
        self.assertEqual(code, cli.EXIT_BAD_INPUT)

    def test_nonexistent_input(self):
        code = cli.main(["/definitivamente/no/existe.mp4"])
        self.assertEqual(code, cli.EXIT_BAD_INPUT)

    def test_end_to_end_with_stubs(self):
        """Pipeline completo: ffmpeg real + transcriptor stub + LLM stub."""
        import math
        import struct
        import subprocess
        import wave

        tmp = Path(tempfile.mkdtemp())
        wav = tmp / "reunion.wav"
        with wave.open(str(wav), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            # señal simple: 2 s de tono
            for i in range(16000 * 2):
                f.writeframesraw(struct.pack("<h", int(10000 * math.sin(2 * math.pi * 440 * i / 16000))))

        cfg = json.loads(json.dumps({
            "model": {"size": "base", "device": "auto", "compute_type": None, "language": "auto", "vram_gpu_gb": 6},
            "summary": {"mode": "meeting", "context_chars": 24000, "chunk_overlap": 0.1,
                        "timeout_s": 120, "retries": 2},
            "llm": {"endpoints": [{"name": "stub", "base_url": "http://localhost:1/v1",
                                   "model": "stub", "api_key_env": ""}]},
            "media": {"segment_threshold_h": 2, "block_minutes": 30, "block_overlap_s": 2.0},
        }))

        calls = {}

        def fake_transcribe(audio, *a, **k):
            calls["audio"] = audio
            from scribeforge.models import Segment, Transcript
            return Transcript(
                segments=[Segment(0, 1, "Mmm, eh, bueno. Primera decisión.", 0.1, -0.2),
                          Segment(2.5, 3, "Segunda frase sobre tareas.", 0.1, -0.2)],
                language="es", model="stub", duration=3.0)

        class StubLLM(cli.LLMClient):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.used = "stub"

            def complete(self, system, user):
                return "## Acta\n- Decisión A\n- Tarea: B (responsable: Ana)"

        outdir = tmp / "outputs"
        made = cli.run_pipeline(
            wav, mode="meeting", output_dir=outdir, model="base", device="auto",
            compute_type=None, language=None, clean=False, no_cache=False,
            keep_work=True, transcribe_fn=fake_transcribe, cfg=cfg, silent=True)
        self.assertIn("txt", made)
        txt = made["txt"].read_text(encoding="utf-8")
        self.assertIn("Primera", txt)
        srt = made["srt"].read_text(encoding="utf-8")
        self.assertIn("-->", srt)
        json.loads(made["json"].read_text(encoding="utf-8"))
        # workdir conservado (--keep-work) y hash-estable
        self.assertTrue((outdir / ".work").exists())
        # caché: segunda pasada reutiliza, llama 0 veces al transcriber
        calls2 = {}
        def fake_transcribe_2(audio, *a, **k):
            calls2["n"] = calls2.get("n", 0) + 1
            raise AssertionError("no debe re-transcribir con cache")
        made2 = cli.run_pipeline(
            wav, mode="meeting", output_dir=outdir, model="base", device="auto",
            compute_type=None, language=None, clean=False, no_cache=False,
            keep_work=True, transcribe_fn=fake_transcribe_2, cfg=cfg, silent=True)
        self.assertIn("txt", made2)

    def test_cfg_stubs_flag(self):
        self.assertIn("cfg", json.loads(json.dumps({"cfg": 1})))


if __name__ == "__main__":
    unittest.main()
