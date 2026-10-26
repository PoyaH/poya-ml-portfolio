# Poya Haghnegahdar — NLP portfolio

Selected NLP experiments developed from hands-on coursework, with clearly labelled AI-assisted reproducibility extensions prepared in October 2026.

I enjoy turning a messy problem into a structured understanding, then testing that understanding through models, numerical experiments and software.

| Project | What is demonstrated |
|---|---|
| [Pizza FAQ intent classification](projects/pizza-intents/README.md) | Executed baseline/neural comparison, fixed splits, saved metrics and Jupyter walkthrough |
| [Statistical autocomplete](projects/autocomplete/README.md) | Conditional next-word probabilities and seven behavioral checks |

## Results and limits

On the pizza dataset's 54-example test split, logistic regression achieved 75.9% accuracy and the neural classifier 70.4%. These are small-sample internal results on an existing course dataset, not independent external validation or production performance.

The autocomplete implementation has synthetic correctness checks, not a held-out corpus benchmark. These new scripts were prepared with AI assistance and are distinct from the original submitted coursework.

## Run

Use Python 3.11. Install the packages in `requirements.txt` in a virtual environment. `environment-tested.txt` records the exact tested macOS environment. Each project README explains execution; the pizza walkthrough displays the included results without access to the original dataset. Rerunning training requires a compatible local intent JSON, which is not distributed here.

Original teaching materials, datasets, personal files, credentials, model weights and unreviewed historical notebooks are excluded. Course-derived transformer modules are also excluded pending attribution/redistribution review.

[Professional portfolio](https://www.datascienceportfol.io/poyah)
