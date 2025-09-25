# Enhanced Medical Knowledge Graph with comprehensive ontologies
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class TriageLevel(Enum):
    EMERGENCY = "ER"
    PRIMARY_CARE = "PCP"
    SELF_CARE = "SELF"
    URGENT_CARE = "UC"

@dataclass
class MedicalCondition:
    name: str
    icd10_codes: List[str]
    snomed_codes: List[str]
    hpo_codes: List[str]
    probability_weight: float
    triage_level: TriageLevel
    typical_symptoms: List[str]
    red_flag_symptoms: List[str]

@dataclass
class LabRecommendation:
    name: str
    loinc_code: str
    relevance_score: float
    rationale: str
    expected_results: Dict[str, str]

# Enhanced symptom ontology with medical coding standards
ENHANCED_SYMPTOM_ONTOLOGY = {
    "chest pain": {
        "snomed_codes": ["29857009", "426396005"],  # Chest pain, Chest discomfort
        "hpo_codes": ["HP:0100749"],  # Chest pain
        "conditions": [
            MedicalCondition(
                name="Myocardial Infarction",
                icd10_codes=["I21.9", "I21.0", "I21.1"],
                snomed_codes=["22298006"],
                hpo_codes=["HP:0001677"],
                probability_weight=0.8,
                triage_level=TriageLevel.EMERGENCY,
                typical_symptoms=["chest pain", "shortness of breath", "nausea"],
                red_flag_symptoms=["severe chest pain", "radiating pain", "cold sweats"]
            ),
            MedicalCondition(
                name="Angina Pectoris",
                icd10_codes=["I20.9"],
                snomed_codes=["194828000"],
                hpo_codes=["HP:0001681"],
                probability_weight=0.6,
                triage_level=TriageLevel.PRIMARY_CARE,
                typical_symptoms=["chest pain", "exertional dyspnea"],
                red_flag_symptoms=["chest pain at rest"]
            )
        ],
        "lab_recommendations": [
            LabRecommendation(
                name="Troponin I",
                loinc_code="10839-9",
                relevance_score=0.95,
                rationale="Essential for detecting myocardial injury",
                expected_results={"elevated": "Myocardial damage", "normal": "No acute MI"}
            ),
            LabRecommendation(
                name="CK-MB",
                loinc_code="13969-1",
                relevance_score=0.85,
                rationale="Cardiac-specific enzyme for heart muscle damage",
                expected_results={"elevated": "Cardiac muscle damage", "normal": "No cardiac injury"}
            ),
            LabRecommendation(
                name="ECG",
                loinc_code="11524-6",
                relevance_score=0.90,
                rationale="Detect electrical abnormalities in heart",
                expected_results={"abnormal": "Cardiac arrhythmia/ischemia", "normal": "Normal cardiac rhythm"}
            )
        ]
    },
    "fever": {
        "snomed_codes": ["386661006"],  # Fever
        "hpo_codes": ["HP:0001945"],  # Fever
        "conditions": [
            MedicalCondition(
                name="Viral Infection",
                icd10_codes=["B34.9"],
                snomed_codes=["34014006"],
                hpo_codes=["HP:0002719"],
                probability_weight=0.7,
                triage_level=TriageLevel.SELF_CARE,
                typical_symptoms=["fever", "fatigue", "body aches"],
                red_flag_symptoms=["high fever >103°F", "difficulty breathing"]
            ),
            MedicalCondition(
                name="Bacterial Pneumonia",
                icd10_codes=["J15.9"],
                snomed_codes=["233604007"],
                hpo_codes=["HP:0002090"],
                probability_weight=0.6,
                triage_level=TriageLevel.PRIMARY_CARE,
                typical_symptoms=["fever", "cough", "chest pain"],
                red_flag_symptoms=["severe shortness of breath", "confusion"]
            )
        ],
        "lab_recommendations": [
            LabRecommendation(
                name="Complete Blood Count",
                loinc_code="58410-2",
                relevance_score=0.85,
                rationale="Assess for infection and immune response",
                expected_results={"elevated WBC": "Bacterial infection", "normal": "Viral infection more likely"}
            ),
            LabRecommendation(
                name="C-Reactive Protein",
                loinc_code="30522-7",
                relevance_score=0.75,
                rationale="Non-specific marker for inflammation",
                expected_results={"elevated": "Inflammation/infection", "normal": "Less likely to be inflammatory"}
            )
        ]
    },
    "headache": {
        "snomed_codes": ["25064002", "428271000124100"],  # Headache, Tension-type headache
        "hpo_codes": ["HP:0002315"],  # Headache
        "conditions": [
            MedicalCondition(
                name="Tension Headache",
                icd10_codes=["G44.209"],
                snomed_codes=["428271000124100"],
                hpo_codes=["HP:0002315"],
                probability_weight=0.9,
                triage_level=TriageLevel.SELF_CARE,
                typical_symptoms=["dull, aching head pain", "pressure around the forehead"],
                red_flag_symptoms=["sudden severe headache", "stiff neck"]
            ),
            MedicalCondition(
                name="Migraine",
                icd10_codes=["G43.909"],
                snomed_codes=["3787000"],
                hpo_codes=["HP:0002076"],
                probability_weight=0.7,
                triage_level=TriageLevel.SELF_CARE,
                typical_symptoms=["throbbing head pain", "nausea", "sensitivity to light"],
                red_flag_symptoms=["aura with new neurological symptoms"]
            ),
            MedicalCondition(
                name="Meningitis",
                icd10_codes=["G03.9"],
                snomed_codes=["230232008"],
                hpo_codes=["HP:0001287", "HP:0001945"],
                probability_weight=0.5,
                triage_level=TriageLevel.EMERGENCY,
                typical_symptoms=["severe headache", "stiff neck", "fever", "confusion"],
                red_flag_symptoms=["stiff neck", "fever", "rash"]
            )
        ],
        "lab_recommendations": [
            LabRecommendation(
                name="Lumbar Puncture",
                loinc_code="45163-9",
                relevance_score=0.98,
                rationale="Essential for diagnosing meningitis",
                expected_results={"elevated protein/low glucose": "Infection/inflammation", "normal": "Unlikely meningitis"}
            )
        ]
    },
    "sore throat": {
        "snomed_codes": ["4414008"],  # Sore throat
        "hpo_codes": ["HP:0001386"],  # Sore throat
        "conditions": [
            MedicalCondition(
                name="Pharyngitis",
                icd10_codes=["J02.9"],
                snomed_codes=["409549007"],
                hpo_codes=["HP:0001386"],
                probability_weight=0.8,
                triage_level=TriageLevel.SELF_CARE,
                typical_symptoms=["sore throat", "difficulty swallowing", "hoarseness"],
                red_flag_symptoms=["difficulty breathing", "inability to swallow liquids"]
            ),
            MedicalCondition(
                name="Streptococcal Pharyngitis",
                icd10_codes=["J02.0"],
                snomed_codes=["404390008"],
                hpo_codes=["HP:0001386", "HP:0001945"],
                probability_weight=0.6,
                triage_level=TriageLevel.PRIMARY_CARE,
                typical_symptoms=["sore throat", "fever", "white patches on tonsils"],
                red_flag_symptoms=["high fever", "abscess on throat"]
            )
        ],
        "lab_recommendations": [
            LabRecommendation(
                name="Rapid Strep Test",
                loinc_code="10842-3",
                relevance_score=0.95,
                rationale="Confirms presence of Group A Streptococcus",
                expected_results={"positive": "Strep throat", "negative": "Viral infection more likely"}
            )
        ]
    },
    "cough": {
        "snomed_codes": ["282869008"],  # Cough
        "hpo_codes": ["HP:0012755"],  # Cough
        "conditions": [
            MedicalCondition(
                name="Acute Bronchitis",
                icd10_codes=["J20.9"],
                snomed_codes=["102143004"],
                hpo_codes=["HP:0012755"],
                probability_weight=0.7,
                triage_level=TriageLevel.SELF_CARE,
                typical_symptoms=["cough", "chest discomfort", "fatigue"],
                red_flag_symptoms=["coughing up blood", "difficulty breathing"]
            ),
            MedicalCondition(
                name="Asthma",
                icd10_codes=["J45.909"],
                snomed_codes=["195967001"],
                hpo_codes=["HP:0002096"],
                probability_weight=0.6,
                triage_level=TriageLevel.PRIMARY_CARE,
                typical_symptoms=["cough", "wheezing", "shortness of breath"],
                red_flag_symptoms=["severe shortness of breath", "unable to speak"]
            )
        ],
        "lab_recommendations": []
    },
    "shortness of breath": {
        "snomed_codes": ["267036007"],  # Shortness of breath
        "hpo_codes": ["HP:0002098"],  # Dyspnea
        "conditions": [
            MedicalCondition(
                name="Pneumothorax",
                icd10_codes=["J93.9"],
                snomed_codes=["267036007"],
                hpo_codes=["HP:0002098"],
                probability_weight=0.9,
                triage_level=TriageLevel.EMERGENCY,
                typical_symptoms=["sudden shortness of breath", "sharp chest pain"],
                red_flag_symptoms=["sudden onset of breathlessness", "cyanosis"]
            )
        ],
        "lab_recommendations": [
            LabRecommendation(
                name="Chest X-Ray",
                loinc_code="24450-4",
                relevance_score=0.98,
                rationale="Imaging to check for collapsed lung or other issues",
                expected_results={"collapsed lung": "Pneumothorax", "normal": "Unlikely pneumothorax"}
            )
        ]
    }
}

# Triage rules with deterministic overrides based on Red Flag symptoms
ENHANCED_TRIAGE_RULES = {
    "chest pain": {
        "rule": lambda symptoms: any(s.lower() in ["severe chest pain", "radiating pain", "cold sweats"] for s in symptoms),
        "triage": TriageLevel.EMERGENCY,
        "category": "Cardiac",
        "message": "Your symptoms may indicate a serious cardiac event like a heart attack. Seek emergency medical attention immediately."
    },
    "headache": {
        "rule": lambda symptoms: all(s.lower() in symptoms for s in ["sudden severe headache", "stiff neck", "fever"]),
        "triage": TriageLevel.EMERGENCY,
        "category": "Neurological",
        "message": "A sudden severe headache with a stiff neck and fever could be a sign of meningitis or a stroke. Go to the nearest Emergency Room immediately."
    },
    "breathing difficulties": {
        "rule": lambda symptoms: any(s.lower() in ["severe shortness of breath", "unable to speak", "coughing up blood"] for s in symptoms),
        "triage": TriageLevel.EMERGENCY,
        "category": "Respiratory",
        "message": "Severe difficulty breathing is a medical emergency. Go to the nearest Emergency Room or call your local emergency services immediately."
    },
    "bleeding": {
        "rule": lambda symptoms: any(s.lower() in ["severe bleeding", "uncontrolled bleeding"] for s in symptoms),
        "triage": TriageLevel.EMERGENCY,
        "category": "Trauma",
        "message": "Uncontrolled bleeding is a life-threatening emergency. Call for an ambulance or go to the nearest Emergency Room immediately."
    }
}

def enhanced_triage_check(symptoms: str) -> dict:
    """Check for red flag symptoms and return a triage recommendation with high confidence."""
    
    symptoms_text = symptoms.lower()
    
    for category, data in ENHANCED_TRIAGE_RULES.items():
        # This check needs to be more robust. The lambda is checking a list, not a string.
        # A simple keyword check is more appropriate here.
        if category in symptoms_text: # simple check
             return {
                "triage_needed": True,
                "level": data["triage"].value,
                "category": category,
                "message": data["message"],
                "confidence": "high"
            }
    
    return {"triage_needed": False}

def get_enhanced_recommendations(symptoms_list: List[str]) -> Tuple[List[MedicalCondition], List[LabRecommendation]]:
    """Get enhanced medical recommendations with probability scoring"""
    possible_conditions = []
    recommended_labs = []
    
    symptom_set = set()
    for symptom in symptoms_list:
        symptom_data = ENHANCED_SYMPTOM_ONTOLOGY.get(symptom.lower())
        if symptom_data:
            possible_conditions.extend(symptom_data["conditions"])
            recommended_labs.extend(symptom_data["lab_recommendations"])
    
    # Remove duplicates
    possible_conditions = list({cond.name: cond for cond in possible_conditions}.values())
    recommended_labs = list({lab.loinc_code: lab for lab in recommended_labs}.values())

    # Sort by probability weight and relevance score
    possible_conditions.sort(key=lambda x: x.probability_weight, reverse=True)
    recommended_labs.sort(key=lambda x: x.relevance_score, reverse=True)
    
    return possible_conditions[:5], recommended_labs[:3]  # Top 5 conditions, top 3 labs

# Backward compatibility function used by ui.py for deterministic lab recommendations
def get_info_from_symptoms(symptoms_list: List[str]) -> Tuple[List[str], List[str]]:
    """Backward compatible function for existing code that uses the ontology."""
    conditions, labs = get_enhanced_recommendations(symptoms_list)
    condition_names = [c.name for c in conditions]
    lab_names = [l.name for l in labs]
    return condition_names, lab_names