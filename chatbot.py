import joblib
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import sys
import os
import contextlib
import io

warnings.filterwarnings("ignore")
""" import sys
import os

nltk_download_output = open(os.devnull, 'w')
sys.stdout = nltk_download_output

# Download required NLTK data
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

sys.stdout = sys._stdout_ """

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the vectorizer and retrieval data
vectorizer = joblib.load('C:\\Users\\Moshe\\Desktop\\chatbot\\models\\vectorizer.pkl')
questions, answers = joblib.load('C:\\Users\\Moshe\\Desktop\\chatbot\\models\\qa_data.pkl')
X = vectorizer.fit_transform(questions)

# Load the generative model
model = torch.load('C:\\Users\\Moshe\\Desktop\\chatbot\\models\\generative_model.pth')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

@contextlib.contextmanager
def suppress_stdout():
    stdout = sys.stdout
    stdout_null = open(os.devnull, 'w')
    sys.stdout = stdout_null
    yield
    sys.stdout.close()
    sys.stdout = stdout

def preprocess_query(query):
    tokens = nltk.word_tokenize(query.lower())
    lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(lemmatized)

def get_retrieval_response(query):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    best_match_index = similarities.argmax()
    
    question = questions[best_match_index] if best_match_index < len(questions) else None
    answer = answers[best_match_index] if best_match_index < len(answers) else None
    
    if question is None or answer is None:
        #print(f"Warning: Retrieved question or answer is None. Index: {best_match_index}")
        return "I couldn't find a specific answer to that question.", "I'm sorry, but I don't have enough information to provide a reliable answer."
    
    return question, answer