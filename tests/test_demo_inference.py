"""CLI and inference wiring checks that do not download model weights."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

from demo import grpo_inference, sft_inference
from demo.inference import build_parser, parse_args, run_inference


ROOT = Path(__file__).resolve().parents[1]


class DemoCliTests(unittest.TestCase):
    def test_help_and_invalid_input_without_site_packages(self):
        for script in ("sft_inference.py", "grpo_inference.py"):
            command = [sys.executable, "-S", str(ROOT / "demo" / script)]
            result = subprocess.run(command + ["--help"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--query", result.stdout)
            self.assertIn("--model", result.stdout)
            for args in ([], ["--query", " "], ["--query", "help", "--temperature", "nan"],
                         ["--query", "help", "--max-new-tokens", "0"],
                         ["--query", "help", "--top-p", "1.1"],
                         ["--query", "help", "--top-k", "-1"]):
                with self.subTest(script=script, args=args):
                    result = subprocess.run(command + args, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("error:", result.stderr)
                    self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_entrypoints_preserve_defaults_and_accept_model_override(self):
        query = 'Show help for "nmap" with {literal braces}.'
        for module, model, temperature, tokens in (
            (sft_inference, "RedSage-K-SFT", 0, 256),
            (grpo_inference, "RedSage-K-SFT-GRPO", 0.1, 8192),
        ):
            with patch.object(module, "run_inference") as run:
                module.main(["--query", query])
                args, messages = run.call_args.args
                self.assertEqual(args.model, "RISys-Lab/" + model)
                self.assertEqual(args.temperature, temperature)
                self.assertEqual(args.max_new_tokens, tokens)
                self.assertIn(query, messages[1]["content"])
                module.main(["--query", query, "--model", "/tmp/local-model", "--seed", "42"])
                args, _ = run.call_args.args
                self.assertEqual(args.model, "/tmp/local-model")
                self.assertEqual(args.seed, 42)

    def test_inference_passes_options_and_prints_only_completion(self):
        for temperature in (0, 0.3):
            with self.subTest(temperature=temperature):
                parser = build_parser("test", "local-model", 256, temperature)
                args = parse_args(parser, ["--query", "Show help", "--dtype", "float32",
                                           "--device-map", "cpu", "--seed", "42"])
                torch = MagicMock()
                transformers = MagicMock()
                tokenizer = transformers.AutoTokenizer.from_pretrained.return_value
                tokenizer.pad_token_id = None
                tokenizer.eos_token_id = 2
                tokenizer.decode.return_value = "<output>nmap --help</output>"
                model = transformers.AutoModelForCausalLM.from_pretrained.return_value.eval.return_value
                inputs = MagicMock()
                inputs.keys.return_value = ["input_ids"]
                inputs.__getitem__.return_value.shape = (1, 7)
                tokenizer.apply_chat_template.return_value.to.return_value = inputs
                output = io.StringIO()
                messages = sft_inference.build_messages(args.query)
                with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
                    with contextlib.redirect_stdout(output):
                        run_inference(args, messages)
                transformers.set_seed.assert_called_once_with(42)
                transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
                    "local-model", torch_dtype=torch.float32, device_map="cpu"
                )
                kwargs = model.generate.call_args.kwargs
                self.assertEqual(kwargs["do_sample"], temperature > 0)
                self.assertEqual(kwargs["pad_token_id"], 2)
                self.assertEqual(kwargs["max_new_tokens"], 256)
                if temperature:
                    self.assertEqual(kwargs["temperature"], temperature)
                    self.assertEqual(kwargs["top_p"], 1.0)
                    self.assertEqual(kwargs["top_k"], 0)
                else:
                    self.assertTrue({"temperature", "top_p", "top_k"}.isdisjoint(kwargs))
                model.generate.return_value.__getitem__.assert_called_once_with((0, slice(7, None)))
                self.assertEqual(output.getvalue(), "<output>nmap --help</output>\n")


if __name__ == "__main__":
    unittest.main()
