\# Quantum Experiments: Q1, Q2, Q3



This folder contains the simulator-based hybrid quantum-classical experiments used in the obesity classification study.



\## Dataset used

The quantum experiments used the top-10 feature version of the processed obesity dataset:

\- `processed\_data/train\_10\_features.csv`

\- `processed\_data/test\_10\_features.csv`



\## Split information

Original ready split:

\- Train: 1669

\- Test: 418



Clean quantum split used in final experiments:

\- Train: 1335

\- Validation: 334

\- Test: 418



\## Quantum models

\### Q1

Pure angle encoding hybrid quantum-classical model:

\- 10 features -> 10 qubits

\- 2 variational layers

\- ring CNOT entanglement

\- all-qubit readout

\- classical head



Final results:

\- Accuracy: 0.7895

\- Macro F1: 0.7769

\- Weighted F1: 0.7822

\- ROC-AUC: 0.9698



\### Q2

Deeper pure angle encoding model:

\- 10 features -> 10 qubits

\- 4 variational layers



Final results:

\- Accuracy: 0.7512

\- Macro F1: 0.7389

\- Weighted F1: 0.7473

\- ROC-AUC: 0.9602



\### Q3

Dual-branch hybrid encoding model:

\- Angle branch: 10 qubits, 2 layers

\- Amplitude branch: 4 qubits, 2 layers

\- Fusion with a classical head



Final results:

\- Accuracy: 0.7871

\- Macro F1: 0.7756

\- Weighted F1: 0.7817

\- ROC-AUC: 0.9705



\## Notes

\- Q1 achieved the best overall accuracy and weighted F1 among the quantum models.

\- Q3 achieved the best ROC-AUC among the quantum models.

\- Q2 showed that increasing circuit depth did not improve performance in this setting.



\## Repository structure

\- `comparison\_tables/` -> final comparison CSV files

\- `circuit\_evidence/` -> circuit definitions, ASCII circuit outputs, and circuit diagrams

\- `experiments/` -> experiment-specific figures, tables, and notes

\- `notebooks/` -> Colab notebooks used for the experiments

