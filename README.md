# MLE-mini-project

This Readme is for all of the important informations around our project.

---

GOAL OF THE PROJECT

"Predict the power generation of all 3 solar power plants for the first 4 months of 2014 given forecasts of 12 different environmental measurements." + beat Benchmark

More information at keggle: https://www.kaggle.com/competitions/machine-learning-energy-2324-mini-project/overview

---

WORKING TASKS
1. Define basic approach
2. Define most promising solution
3. Code & evaluate solution
4. if solution superior to benchmark: DONE

   else: go back to step 2

---

APPROACH 

Data preprocessing:
- normalize features for similar scaling (0-1)
- hot-one encoded features for year / month / day / hours / minutes (or maybe only month / day / hours)
- augment data by adding more features? (examples: squareroot / lagged / (other augmentations) of temperature / solar radiation / (other features)
- add new features by combination of different features
- reduce features? (--> with LASSO regression)
- label test data

Possible solutions:
- Linear Regression (LASSO wegen geringer schnellerer Berechnung durch geringere Anzahl an Features?)
- Neural Network

---

SOLUTION

tbd

