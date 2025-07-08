import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import pickle

# Define your model architecture (same as training)
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# --------------------------- Streamlit UI ---------------------------

st.set_page_config(page_title="Credit Scoring RL Agent", layout="centered")

st.title("💳 Credit Scoring Using RL (DQN Agent)")
st.markdown("Upload borrower data to get Approve/Reject decisions from a trained reinforcement learning agent.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload borrower CSV (without label column)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📋 Borrower Data Preview", df.head())

    # Load feature order
    with open("feature_columns.pkl", "rb") as f:
        feature_cols = pickle.load(f)

    # Reorder columns
    df = df[feature_cols]

    # Load trained model
    model = QNetwork(input_dim=len(feature_cols), output_dim=2)
    model.load_state_dict(torch.load("dqn_credit_model.pth", map_location=torch.device('cpu')))
    model.eval()

    # Predict using agent
    decisions = []
    for i in range(len(df)):
        row = torch.FloatTensor(df.iloc[i].values).unsqueeze(0)
        with torch.no_grad():
            output = model(row)
        action = torch.argmax(output).item()
        decision = "✅ Approve" if action == 1 else "❌ Reject"
        decisions.append(decision)

    # Display results
    df["RL Decision"] = decisions
    st.success("🎉 Borrowers scored successfully!")
    st.write(df)

    # Download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Results", data=csv, file_name="scored_borrowers.csv", mime="text/csv")
