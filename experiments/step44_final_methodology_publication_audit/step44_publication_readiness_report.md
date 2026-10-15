# Step 44 — Final Methodology & Publication-Readiness Audit

**Final status:** RESEARCH PIPELINE READY FOR FINAL PAPER STAGE

## 1. Research Configuration

- Primary dataset: CIC-IDS2018
- Final model: Tuned Random Forest
- Feature count: 30
- Class count: 14
- External dataset: UNSW-NB15
- Explainability: SHAP

## 2. Final Model

The frozen Step-21 Random Forest remains the primary final model based on the completed comparative experiments.

- Validation Accuracy: 0.952812
- Validation Balanced Accuracy: 0.790996
- Validation Macro F1: 0.790499

## 3. Research Integrity

- **training_after_step21:** PASS
- **hyperparameter_optimization_after_step21:** PASS
- **test_set_used_for_model_selection:** PASS
- **unsw_used_for_model_selection:** PASS
- **unsupported_cross_dataset_prediction:** PASS
- **frozen_model_modified:** PASS
- **artificial_feature_semantic_mapping:** PASS
- **temporal_claim_without_timestamp_validation:** PASS

## 4. Methodology Audit

- **Step 21 — controlled RF tuning:** PASS — Controlled hyperparameter optimization using training data and stratified CV.
- **Step 27 — LSTM comparison:** PASS WITH LIMITATION — Comparative deep-learning experiment retained, without claiming validated temporal semantics.
- **Step 30 — timestamp recovery:** PASS — Raw timestamp recovery attempt documented; recovery was insufficient for temporal reconstruction.
- **Step 35 — LightGBM:** PASS — Alternative tabular learner evaluated and shown to be non-competitive.
- **Step 36 — heterogeneous comparison:** PASS — Alternative models compared before final research model consolidation.
- **Step 39 — untouched test evaluation:** PASS — Held-out CIC-IDS2018 test set reserved for final unbiased performance reporting.
- **Step 40 — UNSW external validation:** PASS — External generalization dataset audited without unsupported feature transformation.
- **Step 41 — SHAP:** PASS — Global and class-specific explainability generated from the frozen Random Forest.
- **Step 42 — deployment demonstration:** PASS — Deployment-oriented inference feasibility demonstrated without replacing formal test evaluation.
- **Step 43 — results consolidation:** PASS — Completed experimental evidence consolidated without retraining.

## 5. Publication-Readiness Checklist

- [x] Clear research objective
- [x] Multiclass IDS formulation
- [x] Primary dataset documented
- [x] Training/validation/test separation documented
- [x] Class imbalance documented
- [x] Training-only balancing documented
- [x] Controlled hyperparameter optimization documented
- [x] Multiple model families evaluated
- [x] Macro-level metrics reported
- [x] Balanced Accuracy reported
- [x] Untouched test set preserved
- [x] Explainability analysis completed
- [x] External dataset compatibility audited
- [x] Temporal limitation explicitly documented
- [x] Deployment feasibility demonstrated
- [x] No fabricated external performance
- [x] No unsupported temporal claim

## 6. Important Limitations to Report

- LSTM temporal interpretation must remain limited because Step-30 timestamp recovery was insufficient.
- Step-42 deployment-demo accuracy must not be presented as the formal final test-set performance.
- UNSW-NB15 should be described as an external compatibility/generalization audit, not as a successful numerical external validation result.
- Step-39 contains 4 JSON artifacts. The authoritative final test-result artifact should be identified explicitly before paper writing.

## 7. Critical Issues

No critical methodological issues were detected by this automated audit.

## 8. Frozen Model Integrity

- SHA-256: `3fe363e4dfa6c4f137d5ea7e17e90c40cd202faf483b91e7555e47bc46ba6eb3`

The hash is recorded as a reproducibility reference for the final frozen Random Forest artifact.

## 9. Final Research Position

The completed workflow supports a multiclass network attack detection framework based on CIC-IDS2018, using controlled model development, imbalance-aware training, heterogeneous model comparison, untouched test evaluation, explainability, external-dataset compatibility analysis, and deployment-oriented inference evidence.

The LSTM experiment is retained as a comparative model, but temporal claims are restricted because reliable timestamp reconstruction was not possible for the B1 training data.

UNSW-NB15 is retained as an external/generalization dataset, but no unsupported numerical cross-dataset performance is reported because the frozen CIC-IDS2018 feature schema is incompatible with UNSW-NB15.

The deployment demonstration is supporting evidence of inference feasibility and is not substituted for the formal held-out CIC-IDS2018 test evaluation.

### Next Stage

Step 45 will perform final project closure and freeze the research-engineering evidence for the paper stage.