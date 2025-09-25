import json
import asyncio
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class EvaluationMetrics:
    top_n_accuracy: float
    precision: float
    recall: float
    f1_score: float
    triage_sensitivity: float
    lab_recommendation_precision: float
    brier_score: float
    calibration_error: float

class ClinicalVignetteGenerator:
    """Generate synthetic clinical vignettes for evaluation"""
    
    def __init__(self):
        self.vignettes = [
            {
                "id": "cv_001",
                "symptoms": ["severe chest pain", "shortness of breath", "nausea"],
                "demographics": {"age": 55, "sex": "male", "smoking": True},
                "expected_conditions": ["Myocardial Infarction"],
                "expected_triage": "ER",
                "expected_labs": ["10839-9", "13969-1"],  # Troponin, CK-MB
                "severity": "high"
            },
            {
                "id": "cv_002", 
                "symptoms": ["fever", "cough", "fatigue"],
                "demographics": {"age": 28, "sex": "female", "smoking": False},
                "expected_conditions": ["Viral Infection"],
                "expected_triage": "SELF",
                "expected_labs": ["58410-2"],  # CBC
                "severity": "low"
            },
            {
                "id": "cv_003",
                "symptoms": ["severe headache", "neck stiffness", "fever"],
                "demographics": {"age": 22, "sex": "male", "smoking": False},
                "expected_conditions": ["Meningitis"],
                "expected_triage": "ER", 
                "expected_labs": ["45163-9"],  # Lumbar puncture
                "severity": "high"
            }
        ]
    
    def generate_vignette(self, complexity: str = "medium") -> Dict:
        """Generate a clinical vignette based on complexity level"""
        # In production, this would use GPT-4 to generate diverse scenarios
        return self.vignettes[np.random.randint(0, len(self.vignettes))]

class SymptomCheckerEvaluator:
    """Comprehensive evaluation system for the symptom checker"""
    
    def __init__(self, symptom_checker_api):
        self.api = symptom_checker_api
        self.vignette_generator = ClinicalVignetteGenerator()
        
    async def evaluate_top_n_accuracy(self, test_cases: List[Dict], n: int = 3) -> float:
        """Evaluate Top-N inclusion accuracy"""
        correct_predictions = 0
        total_cases = len(test_cases)
        
        for case in test_cases:
            symptoms = " ".join(case["symptoms"])
            result = await self.api.analyze_symptoms(symptoms)
            
            predicted_conditions = [c["name"] for c in result.get("conditions", [])][:n]
            expected_conditions = case["expected_conditions"]
            
            if any(exp in predicted_conditions for exp in expected_conditions):
                correct_predictions += 1
                
        return correct_predictions / total_cases if total_cases > 0 else 0.0
    
    async def evaluate_triage_sensitivity(self, test_cases: List[Dict]) -> float:
        """Evaluate triage decision sensitivity"""
        correct_triage = 0
        emergency_cases = [case for case in test_cases if case["expected_triage"] == "ER"]
        
        for case in emergency_cases:
            symptoms = " ".join(case["symptoms"])
            result = await self.api.analyze_symptoms(symptoms)
            
            if result.get("triage_level") == "ER":
                correct_triage += 1
                
        return correct_triage / len(emergency_cases) if emergency_cases else 0.0

    async def evaluate_lab_recommendation_precision(self, test_cases: List[Dict]) -> float:
        """Evaluate precision of lab recommendations"""
        all_precisions = []
        
        for case in test_cases:
            symptoms = " ".join(case["symptoms"])
            result = await self.api.analyze_symptoms(symptoms)
            
            predicted_labs = {lab["loinc_code"] for lab in result.get("lab_recommendations", [])}
            expected_labs = set(case["expected_labs"])
            
            if not predicted_labs:
                if not expected_labs:
                    all_precisions.append(1.0) # Correctly predicted no labs
                else:
                    all_precisions.append(0.0) # Failed to predict required labs
                continue

            correct_predictions = len(predicted_labs & expected_labs)
            precision = correct_predictions / len(predicted_labs)
            all_precisions.append(precision)
        
        return np.mean(all_precisions) if all_precisions else 0.0

    async def _evaluate_classification_metrics(self, test_cases: List[Dict]) -> Tuple[float, float, float]:
        """Calculates Precision, Recall, and F1 Score for condition prediction."""
        y_true = []
        y_pred = []

        all_possible_conditions = sorted([
            "Myocardial Infarction", "Angina Pectoris", "Viral Infection", 
            "Bacterial Pneumonia", "Tension Headache", "Migraine", "Meningitis",
            "Pharyngitis", "Streptococcal Pharyngitis", "Acute Bronchitis",
            "Asthma", "Pneumothorax"
        ])
        
        for case in test_cases:
            symptoms = " ".join(case["symptoms"])
            result = await self.api.analyze_symptoms(symptoms)
            
            predicted_conditions = {c["name"] for c in result.get("conditions", [])}
            expected_conditions = set(case["expected_conditions"])

            true_vector = [1 if cond in expected_conditions else 0 for cond in all_possible_conditions]
            pred_vector = [1 if cond in predicted_conditions else 0 for cond in all_possible_conditions]
            
            y_true.append(true_vector)
            y_pred.append(pred_vector)

        precision = precision_score(y_true, y_pred, average='samples', zero_division=0)
        recall = recall_score(y_true, y_pred, average='samples', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='samples', zero_division=0)
        
        return precision, recall, f1

    async def run_comprehensive_evaluation(self, num_tests: int = 100) -> EvaluationMetrics:
        """
        Runs a comprehensive evaluation and returns all metrics.
        """
        test_cases = [self.vignette_generator.generate_vignette() for _ in range(num_tests)]
        
        top_n_accuracy_val = await self.evaluate_top_n_accuracy(test_cases)
        triage_sensitivity_val = await self.evaluate_triage_sensitivity(test_cases)
        lab_precision_val = await self.evaluate_lab_recommendation_precision(test_cases)
        precision_val, recall_val, f1_val = await self._evaluate_classification_metrics(test_cases)
        
        # Placeholder for other metrics
        brier = 0.0
        calibration = 0.0
        
        return EvaluationMetrics(
            top_n_accuracy=top_n_accuracy_val,
            precision=precision_val,
            recall=recall_val,
            f1_score=f1_val,
            triage_sensitivity=triage_sensitivity_val,
            lab_recommendation_precision=lab_precision_val,
            brier_score=brier,
            calibration_error=calibration
        )
        
    def generate_evaluation_report(self, metrics: EvaluationMetrics, num_tests: int) -> str:
        """Generates a markdown report from the evaluation metrics."""
        report = f"""
# AI Health Assistant Evaluation Report
**Date**: {datetime.now().isoformat()}
**Total Test Cases**: {num_tests}

---
##  Diagnostic Accuracy
- **Top-3 Accuracy**: {metrics.top_n_accuracy:.3f}
- **Precision**: {metrics.precision:.3f}
- **Recall**: {metrics.recall:.3f}
- **F1 Score**: {metrics.f1_score:.3f}

---
## Triage Performance
- **Triage Sensitivity (for ER cases)**: {metrics.triage_sensitivity:.3f}
- **Correctly Detected Emergencies**: {metrics.triage_sensitivity * 100:.1f}%

---
### Lab Recommendation Quality
- **Lab Precision**: {metrics.lab_recommendation_precision:.3f}
- **Relevancy of Lab Recommendations**: {metrics.lab_recommendation_precision * 100:.1f}%

---
### Calibration Metrics (Future Work)
- **Brier Score**: {metrics.brier_score:.3f} (lower is better)
- **Calibration Error**: {metrics.calibration_error:.3f} (lower is better)

---
## Recommendations
- **Diagnostic Accuracy**: {'✅ Excellent performance' if metrics.f1_score > 0.8 else '⚠️ Needs improvement'}
- **Triage Safety**: {'✅ Safe triage decisions' if metrics.triage_sensitivity > 0.95 else '⚠️ CRITICAL: Improve emergency detection'}
- **Lab Quality**: {'✅ Relevant lab recommendations' if metrics.lab_recommendation_precision > 0.7 else '⚠️ Refine lab suggestions'}
        """
        return report

# Usage example
async def run_evaluation():
    class MockSymptomCheckerAPI:
        async def analyze_symptoms(self, symptoms: str):
            if "chest pain" in symptoms:
                return {
                    "conditions": [{"name": "Myocardial Infarction", "probability": 0.8}],
                    "triage_level": "ER",
                    "lab_recommendations": [{"loinc_code": "10839-9", "name": "Troponin I"}]
                }
            if "headache" in symptoms and "fever" in symptoms:
                return {
                    "conditions": [{"name": "Meningitis", "probability": 0.6}],
                    "triage_level": "ER",
                    "lab_recommendations": [{"loinc_code": "45163-9", "name": "Lumbar Puncture"}]
                }
            return {
                "conditions": [{"name": "Viral Infection", "probability": 0.8}],
                "triage_level": "SELF",
                "lab_recommendations": [{"loinc_code": "58410-2", "name": "CBC"}]
            }
    
    evaluator = SymptomCheckerEvaluator(MockSymptomCheckerAPI())
    num_tests = 50
    metrics = await evaluator.run_comprehensive_evaluation(num_tests=num_tests)
    report = evaluator.generate_evaluation_report(metrics, num_tests)
    print(report)

if __name__ == "__main__":
    asyncio.run(run_evaluation())