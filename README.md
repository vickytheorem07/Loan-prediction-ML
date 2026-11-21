# Loan Payback Prediction

This project trains a classifier to estimate whether a loan will be paid back. It includes a FastAPI prediction endpoint and a Streamlit form that use the same saved model bundle.

## Prepare the model

1. Install dependencies from the project root:

   ```powershell
   uv sync
   ```
2. Open `Train_model/trainmodel.ipynb` and run the cells in order through **Save model bundle**. This creates `Train_model/model_bundle.pkl`, which is ignored by Git.

## Run FastAPI locally

From the project root:

```powershell
uv run uvicorn main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try `POST /predict`. `GET /health` reports whether the model bundle loaded.

## Run Streamlit locally

From the project root:

```powershell
uv run streamlit run streamlit_app.py
```

The app opens a form for the applicant and loan features, then shows the predicted class, repayment probability, and threshold.

## Run the FastAPI service in Docker

Create the model bundle first, then build and run the image from the project root:

```powershell
docker build -t loan-prediction-api .
docker run --rm -p 8000:8000 loan-prediction-api
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs). The Docker image contains the FastAPI service; Streamlit runs locally with the command above.

This is a learning demo. Do not use the model by itself to make real lending decisions.