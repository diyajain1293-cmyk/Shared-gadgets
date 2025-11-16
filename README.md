# BorrowBox Shared Gadget Library – Advanced Streamlit Dashboard

This app analyses the **Shared Gadget Library / BorrowBox** survey data and provides:

- Market insights with interactive charts
- Classification models (Decision Tree, Random Forest, Gradient Boosting)
- Prediction on new uploaded data
- Customer segmentation using K-Means clustering
- Association Rule Mining for equipment interests

## Files

- `app.py` – main Streamlit app
- `requirements.txt` – Python dependencies (no version pins)
- `Shared_Gadget_Library_Survey_Synthetic_Data.csv` – survey data (add your file here)

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Streamlit Cloud, add these files to a GitHub repo and point the app to `app.py`.
