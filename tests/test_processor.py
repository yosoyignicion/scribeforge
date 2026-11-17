"""Smoke Fase 1: FFmpeg real (tono sintético, sin red, sin modelo)."""
import math
import subprocess
import tempfile
import unittest
from pathlib import Path

from scribeforge import processor


def make_wav(path: Path, seconds: float = 2.0, freq: float = 440) -> None:
    """WAV 16k mono vía ffmpeg (sine + silencio final de 1 s)."""
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}",
         "-af", "apad=pad_dur=1", "-ar", "16000", "-ac", "1", str(path)],
        check=True, capture_output=True)


class TestProcessor(unittest.TestCase):
    def setUp(self):
        processor.check_ffmpeg()  # si falla aquí, el entorno no tiene ffmpeg: fallo legítimo
        self.tmp = Path(tempfile.mkdtemp())
        self.wav = self.tmp / "tone.wav"
        make_wav(self.wav)

    def test_probe_duration(self):
        dur = processor.probe_duration(self.wav)
        self.assertAlmostEqual(dur, 3.0, delta=0.2)

    def test_sha256_stable(self):
        self.assertEqual(processor.sha256_file(self.wav), processor.sha256_file(self.wav))

    def test_extract_audio_16k_mono(self):
        workdir = self.tmp / "work"
        out = processor.extract_audio(self.wav, workdir)
        self.assertTrue(out.exists())
        self.assertEqual(out.name, "audio.wav")
        # 16 kHz mono PCM s16: byte rate ≈ 16000*2 = 32000 B/s
        dur = processor.probe_duration(out)
        self.assertAlmostEqual(dur, 3.0, delta=0.2)
        self.assertGreater(out.stat().st_size, 3.0 * 30000)

    def test_validate_input(self):
        with self.assertRaises(processor.ProcessorError):
            processor.validate_input(self.tmp / "noexiste.mp4")
        with self.assertRaises(processor.ProcessorError):
            processor.validate_input(self.tmp / "contrato.pdf")

    def test_audio_blocks_short_returns_single(self):
        dur = processor.probe_duration(self.wav)
        self.assertEqual(processor.audio_blocks(self.wav, dur), [(self.wav, 0.0)])


if __name__ == "__main__":
    unittest.main()
