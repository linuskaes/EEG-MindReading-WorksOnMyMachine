"""Execute the jury's default path, with no raw data or training dependencies."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_jury_run_all():
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(ROOT / "JURY_WALKTHROUGH.ipynb", as_version=4)
    NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbformat.write(notebook, ROOT / "outputs/jury/executed.ipynb")
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in cells)
    assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
    figures = [output for cell in cells for output in cell.outputs if "image/png" in output.get("data", {})]
    assert len(figures) >= 7
    report = json.loads((ROOT / "outputs/jury/replay/run.json").read_text())
    assert report["mode"] == "replay"
    assert all(row["ranking_entries_changed"] == 0 for row in report["versus_locked_predictions"].values())
    audit = json.loads((ROOT / "outputs/jury/decision-audit.json").read_text())
    assert audit["selected"] == {"audio": .2, "video": .1, "internal": .7, "csls_k": 1}
    assert not (ROOT / "outputs/jury/train").exists()
    assert not (ROOT / "outputs/jury/infer").exists()
    assert not (ROOT / "outputs/jury/retrain-external").exists()
