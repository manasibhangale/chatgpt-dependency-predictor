ChatGPT Dependency Predictor
This is a machine learning-based web app built with Streamlit to predict ChatGPT dependency among students based on their usage patterns and other factors.

FEATURES:
Predicts the likelihood of a user being dependent on ChatGPT
Uses multiple input features such as usage frequency, duration, confidence, and department
Simple and intuitive user interface with Streamlit
Trained machine learning model saved as a .pkl file

INSTALLATION:
1) Clone this repository:
git clone https://github.com/manasibhangale/chatgpt-dependency-predictor.git
cd chatgpt-dependency-predictor
2) Create and activate a virtual environment (optional but recommended):
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
3) Install dependencies:
pip install -r requirements.txt
4)Run the Streamlit app locally: streamlit run app.py
This will open a local web page where you can input features and get the prediction.

DATASET
The dataset used for training the model is included as chatgpt_dependency_dataset.csv.

CONTRIBUTING
Contributions and suggestions are welcome! Feel free to open an issue or submit a pull request.
