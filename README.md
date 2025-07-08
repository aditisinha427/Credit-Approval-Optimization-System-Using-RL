# Credit-Approval-Optimization-System-Using-RL
A Deep Q-Learning Agent that learns optimal credit approval policies by interacting with borrower data, optimizing profits while minimizing defaults.

## Problem Statement

Traditional credit scoring systems rely on static models with fixed assumptions. This project reframes credit risk assessment as a **sequential decision-making problem**, where an agent learns **optimal approval policies** based on past borrower behaviors and repayment patterns.

---

## Tech Stack

- **Python**
- **PyTorch** (for DQN implementation)
- **Pandas, NumPy, Scikit-learn** (for preprocessing, metrics)
- **Streamlit** (for interactive dashboard)
- **Matplotlib / Seaborn** (for reward curves, EDA)

---

## Dataset

Dataset Source: [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  
Rows: 30,000 | Columns: 6 (final features after preprocessing)

### Selected Features:
- `LIMIT_BAL` - Credit limit
- `AGE_GROUP` - Categorized as 0 (<30), 1 (30–50), 2 (>50)
- `DTI` - Debt-to-Income ratio
- `AVG_BILL_AMT` - Average bill amount (past 6 months)
- `AVG_PAY_AMT` - Average repayment amount (past 6 months)
- `OVERDUE_COUNT` - Count of overdue months

---
Features:
- Upload borrower CSV (with same feature format)
- See live RL predictions (`Approve` / `Reject`)
- Download predictions as CSV

To run:

```bash
streamlit run app.py
