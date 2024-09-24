from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing. You can specify specific origins as needed.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Define the model input structure
class ModelInput(BaseModel):
    title: str
    author: str
    text: str

# Load the saved model and vectorizer
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Preprocess text function
def preprocess_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = re.sub(r'\d+', '', text)      # Remove numbers
    text = text.lower()                  # Convert to lowercase
    tokens = text.split()                # Tokenize
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    porter = PorterStemmer()
    tokens = [porter.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)  # Join tokens back into string

@app.post('/fake_news_prediction')
def fake_news_prediction(input_parameters: ModelInput) -> Dict[str, str]:
    # Combine title, author, and text
    combined_text = input_parameters.title + ' ' + input_parameters.author 
    # combined_text = input_parameters.title + ' ' + input_parameters.author + ' ' + input_parameters.text
    
    # Preprocess the combined text
    processed_input = preprocess_text(combined_text)
    
    # Vectorize the preprocessed text
    input_vector = vectorizer.transform([processed_input])
    
    # Make a prediction
    prediction = model.predict(input_vector)
    
    if prediction[0] == 0:
        return {"result": "The NEWS is Real"}
    else:
        return {"result": "The NEWS is Fake"}
