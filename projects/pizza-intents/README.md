# Pizza FAQ intent classification

A reproducible 2026 extension of a course final project. The historical work explored bag-of-words neural classifiers, lemmatization, LSTMs, BART and an unfinished BERT attempt. This edition makes a bounded comparison of a simple logistic-regression baseline with a neural bag-of-words classifier.

## Run

From the repository root, using the dedicated environment:

```sh
source .venv/bin/activate
python projects/pizza-intents/run_experiment.py --data review/pizza-supporting-files/intents_pizza_complete2.json
```

Or open `Walkthrough.ipynb` and select **Python (ML Portfolio)**. The notebook displays the saved result by default; training requires explicitly running its rerun cell. Data stays local pending provenance/redistribution review. The tested environment is recorded in `environment-tested.txt`; it is a snapshot of this macOS environment, not a cross-platform compatibility guarantee.

## Design

- 267 examples across 10 intents; 159 training, 54 validation, 54 test examples.
- Fixed stratified split, seed 42; vocabulary fitted on training examples only.
- Binary bag-of-words features shared by both models.
- Logistic regression with fixed settings; neural architecture 64 units, dropout 0.2, 128 units, softmax output.
- Neural model trained for 50 epochs; epoch selected using minimum validation loss, then test evaluated.
- Exact split indices, dataset checksum, package versions, validation history and per-class reports saved in `results/results.json`.

## Results from this run

| Model | Test accuracy | Macro-F1 |
|---|---:|---:|
| Logistic regression | 75.9% | 0.722 |
| Neural bag-of-words | 70.4% | 0.686 |

The simpler baseline performs better on this split. These results do not establish a general ranking of models. The existing course dataset has already been explored; this is an internal comparison, not fresh external confirmation. The small test set and possible semantic similarity between phrases limit generalization. No out-of-domain rejection performance or deployment capability has been established.

## Historical versus new work

Original notebooks are unchanged. The new script was prepared with AI assistance in October 2026 to improve reproducibility and evaluation, and should not be represented as the original submitted code. It does not reproduce every historical model, lemmatization variant or reported score. Historical course notebooks are not included in this repository. Supplied course scaffolding and data attribution remain under review.
