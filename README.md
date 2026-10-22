# CS 271 Assignment 3: Support Vector Machines

Assignment 3 for CS 271 (Topics in Machine Learning, Prof. Mark Stamp, San José State University, Fall 2025). It covers SVM-based malware classification and a from-scratch implementation of sequential minimal optimization.

## Contents

| File | What it does |
|------|--------------|
| `src/Q_4_12.py` | Trains a linear SVM (scikit-learn) on malware vs. benign samples scored by three detectors (HMM, SSD, OGS), then applies recursive feature elimination to see which scores matter |
| `src/Q_4_15.py` | Implements simplified SMO (SSMO) by hand: pairwise Lagrange multiplier updates, bias recomputation, and the resulting separating hyperplane, with systematic vs. random pair selection |
| `output_Q_4_12.txt`, `output_Q_4_15.txt`, `output_Q_4_15_plot.png` | Captured outputs and the decision-boundary plot |
| `Q_4.1_4.4_4.8ab.pdf` | Written answers for the theory problems |

## Running

```sh
pip install numpy scikit-learn matplotlib
cd src
python3 Q_4_12.py
python3 Q_4_15.py
```
