"""Train once on the bundled dataset and save everything the app needs to start instantly.
Run at image-build time:  python -m app.pretrain"""
import joblib
from pathlib import Path
from .experiment import load_dataset, run_experiments

ROOT = Path(__file__).parent
texts, y, _ = load_dataset((ROOT / "data" / "dataset.txt").read_text(encoding="utf-8"))
results, trained, summary = run_experiments(texts, y)
pool = [(texts[i], int(y[i])) for i in summary.pop("test_idx")]
joblib.dump({"results": results, "trained": trained, "summary": summary, "pool": pool}, ROOT / "data" / "pretrained.joblib")
print(f"pretrained {len(results)} experiments; best accuracy {max(r['accuracy'] for r in results):.3f}")
