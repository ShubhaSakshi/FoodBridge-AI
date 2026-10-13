# FoodBridge AI 🥗

**AI-powered surplus food rescue coordination | SDG 2: Zero Hunger**

## Overview

FoodBridge AI is a Streamlit-based prototype that helps coordinate surplus food donations between food donors and community organisations. It combines a machine-learning demonstration model, an explainable recipient-matching heuristic, and Google's Gemini API to prepare pickup-message drafts.

## Key Features

- **Food listing:** Record food category, quantity, pickup locality, deadline, and handling notes.
- **ML-based urgency estimate:** Uses a trained Random Forest model for an experimental priority prediction.
- **Recipient matching:** Ranks illustrative recipient profiles using food category, capacity, distance, and demo verification status.
- **Gemini integration:** Drafts pickup enquiries for human review.
- **Dashboard:** Displays listing and quantity summaries.
- **Human review:** Requires people to check recipient identity, food suitability, allergens, transport, and pickup arrangements.

## Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib
- Google Gemini API

## Run Locally

1. Install Python.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Configure your Gemini API key in `.streamlit/secrets.toml`:

   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```

4. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

## Machine Learning Note

The current Random Forest model was trained on synthetic demonstration data. Its reported holdout accuracy describes performance on that synthetic dataset only; it is not validated real-world performance. The urgency prediction is experimental and must not be used to determine food safety.

## Limitations and Responsible Use

- Recipient profiles, capacities, distances, and verification indicators are illustrative demo data.
- Recipient ranking is heuristic-based, not an ML prediction.
- Gemini generates draft text only; the app does not send messages or confirm pickups.
- Human verification is mandatory before any real-world donation or pickup.
- No real-world impact or food-safety certification is claimed.

## Future Scope

- Integrate a documented real-world dataset.
- Compare suitable ML models using proper evaluation.
- Develop a verified recipient directory.
- Add secure, consent-based communication and real pickup tracking.

**Project status:** Academic prototype for demonstration and further development.