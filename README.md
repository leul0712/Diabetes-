This repository contains a modular and robust data preprocessing pipeline for the Diabetes Dataset. The primary goal is to transform the raw dataset, which suffers from severe data quality issues (implicit missing values and class imbalance), into a clean, scaled, and balanced format ready for machine learning model training. The project follows professional software engineering practices by separating the core logic into a reusable source directory (src) and dedicating the notebook for execution and visualization.

Pipeline Phases

The data preparation process is divided into five distinct, sequential phases:

Phase 1: Data Collection & Initial Analysis: Load the raw data and perform initial data quality checks.

Phase 2: Data Cleaning: Treat implicit missing values (zeros) using Median Imputation and constrain extreme outliers using IQR Capping.

Phase 3: Data Transformation: Perform Feature Engineering (binning), One-Hot Encoding, and Standard Scaling.

Phase 4: Data Reduction (Exploratory): Analyze feature correlation (Heatmap) and variance retention (PCA) to inform model decisions.

Phase 5: Data Imbalance Handling: Apply the Synthetic Minority Over-sampling Technique (SMOTE) to achieve a 1:1 class ratio.
