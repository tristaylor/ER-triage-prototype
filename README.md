# ER-triage-prototype
AI-assisted ER triage prototype predicting short-term patient deterioration for dynamic prioritization

# Overview
This repository contains a prototype for an **AI-assisted emergency room triage system**. 
The system predicts the probability of **short-term clinical deterioration** for incoming patients and dynamically prioritizes the ER queue to reduce time-to-treatment for high-risk cases.

This project is designed as a **research & prototype tool**, using publicly available data from **MIMIC-IV**, and demonstrates a pipeline from **feature extraction → model training → risk prediction → prioritization → simulation

# Features
- Preprocessing of patient vitals and demographics
- Feature engineering (e.g., shock index, low O₂ flag)
- Label creation for clinical deterioration within 6 hours
- Predictive modeling with XGBoost
- Prioritization engine to rank patients by risk
- ER simulation module to evaluate model impact
- Optional Streamlit UI for visualization

# Data
Data is **not included** due to privacy regulations.  
We use **MIMIC-IV** (or other public ICU datasets) for development and testing.  

Expected input features (can be extracted from MIMIC-IV):

- Heart rate, blood pressure, respiratory rate, oxygen saturation, temperature
- Age, sex
- ICU transfer / ventilation / death (for labels)
