import json
import logging
from pathlib import Path
from src.config import REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class Evaluator:
    """Generates unified model comparison reports and performance metrics."""

    def __init__(self, reports_dir=REPORTS_DIR):
        self.reports_dir = Path(reports_dir)

    def load_results(self):
        clf_file = self.reports_dir / "classification_results.json"
        reg_file = self.reports_dir / "regression_results.json"

        clf_data = {}
        reg_data = {}

        if clf_file.exists():
            with open(clf_file, "r", encoding="utf-8") as f:
                clf_data = json.load(f)

        if reg_file.exists():
            with open(reg_file, "r", encoding="utf-8") as f:
                reg_data = json.load(f)

        return clf_data, reg_data

    def generate_markdown_report(self) -> str:
        clf_data, reg_data = self.load_results()

        md = []
        md.append("# Model Evaluation & Benchmark Report")
        md.append("## Real Estate Investment Advisor — Sanitized Leakage-Free Audit\n")

        # Classification Table
        md.append("### 1. Classification Performance (Target: High_Investment_Potential)")
        md.append("| Model Candidate | Accuracy | Precision | Recall | F1-Score | ROC-AUC |")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
        if "comparison" in clf_data:
            for name, d in clf_data["comparison"].items():
                m = d["metrics"]
                md.append(f"| **{name}** | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | **{m['f1_score']:.4f}** | {m['roc_auc']:.4f} |")
        md.append(f"\n*Champion Classification Model: **{clf_data.get('best_model_name', 'N/A')}***\n")

        # Regression Table
        md.append("### 2. Regression Performance (Target: Price_in_Lakhs)")
        md.append("| Model Candidate | MAE (Lakhs) | RMSE (Lakhs) | R² Score |")
        md.append("| :--- | :---: | :---: | :---: |")
        if "comparison" in reg_data:
            for name, d in reg_data["comparison"].items():
                m = d["metrics"]
                md.append(f"| **{name}** | {m['mae']:.4f} | {m['rmse']:.4f} | **{m['r2_score']:.4f}** |")
        md.append(f"\n*Champion Regression Model: **{reg_data.get('best_model_name', 'N/A')}***\n")

        # Top Features
        md.append("### 3. Top Feature Importances (Sanitized Champion Models)")
        if "top_feature_importances" in clf_data:
            md.append("\n**Classification Top Predictors:**")
            for feat, val in list(clf_data["top_feature_importances"].items())[:7]:
                md.append(f"- `{feat}`: {val:.4f}")

        if "top_feature_importances" in reg_data:
            md.append("\n**Regression Top Predictors:**")
            for feat, val in list(reg_data["top_feature_importances"].items())[:7]:
                md.append(f"- `{feat}`: {val:.4f}")

        # Methodology Note
        md.append("\n### 4. Target Definition and Dataset Limitations Disclosure")
        md.append("- **Zero Observed Historical Labels**: The source dataset contains no ground-truth `Good_Investment` outcome column or longitudinal 5-year future prices.")
        md.append("- **Leakage Elimination**: `Price_per_SqFt`, `infra_score`, and `investment_score_raw` were strictly excluded from model feature matrices.")
        md.append("- **Empirical Valuation Finding**: When leaked derived variables like `Price_per_SqFt` are removed, the raw dataset demonstrates that listing prices were synthetically generated independently of property attributes ($R^2 \\approx 0.000$). Documenting this limitation is methodologically essential rather than presenting artificially inflated metrics.")

        report_content = "\n".join(md)
        report_path = self.reports_dir / "MODEL_BENCHMARKS.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info(f"Report saved to {report_path}")
        return report_content

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.generate_markdown_report()
    print("Report generated.")
