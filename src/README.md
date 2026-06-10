# Scripts

Supporting scripts. Each resolves paths relative to the repository root
(`ROOT = Path(__file__).resolve().parents[1]`) and is meant to be run from there.
Randomness is seeded (`RANDOM_STATE = 42`).

| Script | What it does |
|---|---|
| `execute_notebooks.py` | Runs the notebooks in order, in place. |
| `generate_project_artifacts.py` | Regenerates the notebooks (and the dataset context doc) from code; this is the source of truth for the notebooks. |
| `tune_neural_network.py` | Hyperparameter search for the classifier (width, dropout, weight decay, batch norm, label smoothing); prints a metrics table. |

```bash
python src/execute_notebooks.py
python src/tune_neural_network.py
```
