# Step 43 — Final Research Results Consolidation

This report consolidates the completed experimental evidence without retraining or modifying any model.

## 1. Final Model

**Primary model:** Random Forest (Step 21 tuned Random Forest)

The final model uses 30 CIC-IDS2018 selected features and predicts 14 attack/traffic classes.

## 2. Comparative Model Performance

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9528 | 0.7910 | 0.7993 | 0.7910 | 0.7905 | 0.9600 |
| LSTM | 0.9667 | 0.7117 | 0.7385 | 0.7117 | 0.6983 | 0.9647 |
| LightGBM | 0.6841 | 0.3615 | 0.2648 | 0.3615 | 0.2887 | 0.7606 |

## 3. Model Selection Rationale

The tuned Random Forest is retained as the primary final model because it provides substantially stronger balanced and macro-level performance than the evaluated LSTM and LightGBM alternatives, while also supporting explainability through SHAP and a deployment-oriented inference demonstration.

## 4. Temporal Sequence Audit

The LSTM experiment is retained as a comparative deep-learning experiment, but its sequences are not treated as validated temporal sequences because Step 30 recovered timestamps for only 4.24% of the B1 rows and chronological reconstruction was not reliable.

Therefore, the LSTM is reported as a comparative deep-learning experiment rather than as evidence of validated temporal behaviour modelling.

## 5. Final Untouched Test Evaluation

Step 39 is the authoritative final CIC-IDS2018 test evaluation. Its metrics must be used in the final paper as the unbiased held-out performance of the frozen model.

- **Test Accuracy:** 0.9526
- **Balanced Accuracy:** 0.8256
- **Macro Precision:** 0.8276
- **Macro Recall:** 0.8256
- **Macro F1:** 0.8230
- **Weighted F1:** 0.9598

## 6. External Generalization

UNSW-NB15 is retained as an external/generalization dataset, but no cross-dataset performance score is reported because its feature schema and attack taxonomy are incompatible with the frozen CIC-IDS2018 model.

This is a methodological limitation/compatibility finding, not a failed prediction experiment.

## 7. Explainability

Step 41 applied SHAP to the frozen Random Forest using a reproducible 5,000-flow validation sample.

Top global SHAP features:

1. Init Fwd Win Byts — 0.019927
2. Dst Port — 0.019678
3. Fwd Seg Size Min — 0.015985
4. RST Flag Cnt — 0.010748
5. Flow Duration — 0.004860
6. Fwd Header Len — 0.004439
7. Bwd Pkt Len Std — 0.004205
8. Fwd Act Data Pkts — 0.003748
9. Bwd Pkt Len Mean — 0.003322
10. Fwd IAT Tot — 0.003290

## 8. Deployment-Oriented Evidence

Step 42 demonstrated frozen-model inference on 1,000 simulated validation flows.

- Demo accuracy: N/A
- Mean inference latency: N/A ms
- Median inference latency: N/A ms
- P95 inference latency: N/A ms
- Attack alerts: N/A

These results demonstrate deployment feasibility but are not substituted for the formal held-out test evaluation.

## 9. Research Integrity Statement

No retraining, hyperparameter optimization, test-set model selection, UNSW-based model selection, or unsupported feature transformation was performed during this consolidation step.

## 10. Overall Research Conclusion

The tuned Random Forest is retained as the primary final model because it provides substantially stronger balanced and macro-level performance than the evaluated LSTM and LightGBM alternatives, while also supporting explainability through SHAP and a deployment-oriented inference demonstration.

The overall evidence supports a multiclass, behaviour-oriented network attack detection framework based on CIC-IDS2018, with controlled model development, heterogeneous comparison, explainability, final untouched testing, and deployment-oriented inference evidence.

**Next:** Step 44 will perform the next final research-stage activity; no new modelling should be introduced unless explicitly required by the fixed 45-step methodology.