# Statistical autocomplete

A 2026 AI-assisted rewrite of the course n-gram/Markov-chain experiment. The original Enron notebook is preserved locally. This small implementation corrects conditional-probability normalization, includes the final valid transition, avoids transitions between separate documents, and handles unseen contexts without crashing.

Run `python projects/autocomplete/check_model.py` from the repository root. Seven synthetic correctness cases pass; see checks.json. No held-out Enron prediction-quality result is claimed. The corpus is not distributed.

The course progression also included GPT-2 completion and transformer architecture exercises. Those are separate experiments; this implementation is statistical, not a transformer.
