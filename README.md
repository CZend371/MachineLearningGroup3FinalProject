# MachineLearningGroup3FinalProject
# Student Depression Prediction – SEIS 763-03

**Project goal:** build and evaluate a machine-learning pipeline that predicts whether a student is experiencing depression, based on academic-, lifestyle-, and stress-related factors.  
The work fulfils the semester-long class-project requirements (data ≥ 3 000 rows, multiple ML models, dimensionality reduction, k-fold CV, ensemble, IEEE paper, 20-min presentation).

## 🚀 Quick start

```bash
git clone https://github.com/<org-or-user>/student-depression-ml.git
cd student-depression-ml
conda env create -f environment.yml      # RAPIDS + scikit-learn + imbalanced-learn
conda activate seis763
jupyter lab
---

Open notebook.ipynb and run all cells (GPU recommended, but CPU fallback works—just slower).

📂 Repository layout
student-depression-ml/
│
├── data/
│   └── student_depression_dataset.csv     # original dataset (27 901 rows)
│
├── notebook.ipynb            # full EDA + preprocessing + models + CV
├── models/
│   ├── best_knn.pkl          # tuned KNN (LDA-reduced) pipeline
│   └── best_svm.pkl          # tuned SVM pipeline
│
├── utils/                    # helper modules (will grow)
│   ├── model_svm.py
│   └── model_knn.py
│
├── presentation/             # ← PowerPoint lives here
│   └── draft-deck.pptx
│
├── docs/                     # PDFs for grading
│   ├── project_proposal.pdf
│   ├── project_guide.pdf
│   └── notebook.pdf          # static export
│
├── environment.yml           # conda spec (CPU) + rapids.yaml (GPU optional)
└── README.md

🧠 Methodology (high level)
Cleaning & EDA – handle outliers, standardise numeric features, one-hot encode categoricals.

Class balancing – ADASYN over-sampling + random under-sampling (original split 58 / 42).

Dimensionality reduction – PCA, Kernel PCA, and LDA (best for KNN).

Models tested

David → K-Nearest Neighbours, Support Vector Machine

Karryn → Logistic Regression, Naïve Bayes

Cristian → Random Forest, Decision Tree

Hyper-parameter tuning – 10-fold Stratified CV, GridSearchCV (dask-ml when on GPU).

Evaluation – F1, precision, recall, confusion matrices, ROC-AUC on held-out 10 % test set.

Ensemble (TBD) – soft voting of each teammate’s top model.

Current best single model: linear-kernel SVM (C = 0.1) – F1 = 0.871 on the test set (10-fold CV: 0.871 ± 0.004).
KNN (k = 11, LDA-1D) is ~1 pt behind.

👥 Team

Name	Role & focus	GitHub	Contact
David J. Braun	Pre-processing · KNN & SVM owner · GPU optimisation	@DavidBraun777	davidjbraun777@gmail.com
Karryn J. Leake	Logistic Regression · Naïve Bayes · Report editing	@kleake	leak3729@stthomas.edu
Cristian A. Zendejas	Random Forest · Decision Tree · Ensemble integration	@czendejas	zend7089@stthomas.edu
⏰ Project roadmap & status

Milestone	Target date	Status
Data cleaning & EDA	19 Apr	✅
Dimensionality-reduction prototype	20 Apr	✅
Hyper-param tuning (all six models)	27 Apr	⬜ in progress
Model freeze & k-fold CV	02 May	⬜
Ensemble voting classifier	03 May	⬜
Slide deck draft (20 min)	04 May	⬜
Presentation dry-run	04 May	⬜
Final IEEE paper (6-8 pp)	13 May	⬜
Zip submission (report + code + data)	14 May	⬜
📝 To-do list (open issues)
 Finish grid-searches for Logistic Regression, Naïve Bayes, Random Forest, Decision Tree

 Build ensemble – soft voting vs. stacking; measure lift over SVM

 Add SHAP notebook for interpretability

 Polish PowerPoint deck – high-res plots, flow diagram, rehearse script

 Write IEEE paper – insert final metrics & figures

 Finalise environment.yml with exact versions

 Create GitHub release + Zenodo DOI

Track progress via GitHub Projects.

🤝 Contributing
Teammates: branch naming topic/<short-desc>, open a PR, request one review.
External PRs welcome after the course deadline.

📜 License
Code: MIT. 
Dataset: see /data/LICENSE (original Kaggle terms).


