# Stock Price Prediction Project  

This project uses machine learning to predict stock prices based on historical market data.  
It includes **data preprocessing, model training, evaluation, and inference** – all in a single, reproducible workflow.  

## 📂 Project Structure  



project-root/
│
├── data/ # Raw and processed datasets
├── models/ # Saved models and scalers
├── notebooks/ # Main Jupyter notebook for training
├── src/ # (Optional) Python scripts if modularized
├── environment.yml # Conda environment file (recommended)
└── README.md # This file


## ⚡ Features  

- **Automated Dataset Download** – Fetches data directly from Kaggle using `kagglehub`.  
- **Preprocessing Pipeline** – Scales features with `MinMaxScaler` so models behave nicely.  
- **Model Training** – Trains regression models to predict closing stock prices.  
- **Evaluation Metrics** – Tracks model performance on train/test splits.  
- **Model Saving** – Stores trained models (`.joblib` / `.h5`) for easy reuse.  
- **Inference Script** – Load a saved model and run predictions without retraining.  

## 🛠 Installation  

Clone the repository:  

```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction


Create and activate a Conda environment:

conda create -n stockpred python=3.9 -y
conda activate stockpred


Install dependencies:

pip install -r requirements.txt


💡 If you are using Kaggle datasets, make sure your Kaggle API key is configured (~/.kaggle/kaggle.json).

📊 Usage
1. Train the Model

Open the notebook in Jupyter or VS Code:

jupyter notebook notebooks/stock_prediction.ipynb


Run all cells – this will:

Download the dataset

Preprocess the data

Train the model

Save trained models and scalers in the models/ folder

2. Run Inference

Use the provided inference script:

python src/inference.py --input sample_data.csv --output predictions.csv


This loads the trained model and generates predictions for your input data.

📈 Results

The notebook prints:

Training vs. Test performance metrics

Model accuracy/error percentage

Visualizations of predicted vs. actual prices

This helps you understand how well the model is performing before putting it into production.

🚀 Future Improvements

Add more advanced models (LSTM, Transformer-based predictors)

Hyperparameter tuning with Optuna

Backtesting on multiple tickers

CI/CD pipeline for automatic retraining

📜 License

This project is open source under the MIT License
.
