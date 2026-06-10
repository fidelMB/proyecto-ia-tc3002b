import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [
    ROOT / "notebooks" / "01_eda.ipynb",
    ROOT / "notebooks" / "02_cleaning_feature_engineering.ipynb",
    ROOT / "notebooks" / "03_regression.ipynb",
    ROOT / "notebooks" / "04_pca.ipynb",
    ROOT / "notebooks" / "05_neural_network_classification.ipynb",
    ROOT / "notebooks" / "06_clustering.ipynb",
]


def configure_local_jupyter_dirs():
    dirs = {
        "JUPYTER_CONFIG_DIR": ROOT / ".jupyter" / "config",
        "JUPYTER_DATA_DIR": ROOT / ".jupyter" / "data",
        "JUPYTER_RUNTIME_DIR": ROOT / ".jupyter" / "runtime",
        "IPYTHONDIR": ROOT / ".ipython",
        "MPLCONFIGDIR": ROOT / ".matplotlib",
    }
    for env_name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(path)


def execute_notebook(path: Path):
    print(f"Executing {path.relative_to(ROOT)}")
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    nbformat.write(nb, path)
    print(f"Finished {path.relative_to(ROOT)}")


def main():
    configure_local_jupyter_dirs()
    for notebook in NOTEBOOKS:
        execute_notebook(notebook)
    print("All notebooks executed successfully.")


if __name__ == "__main__":
    main()
