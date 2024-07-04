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
def get_retrieval_confidence(query):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, X).flatten()
    return similarities.max()

def generate_response(query, context=None):
    if context:
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"Question: {query}\nAnswer:"
    
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with suppress_stdout():
        outputs = model.generate(inputs, max_length=1000, num_return_sequences=1, no_repeat_ngram_size=2)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

def ask_for_clarification(query):
    print(f"I'm not sure I fully understand. Can you provide more context about '{query}'?")
    return input("Your clarification (or press Enter to skip): ")

def get_response(query):
    preprocessed_query = preprocess_query(query)
    retrieval_confidence = get_retrieval_confidence(preprocessed_query)
    #print(f"Retrieval confidence: {retrieval_confidence}")

    if retrieval_confidence > 0.8:
        question, response = get_retrieval_response(preprocessed_query)
        #print(f"Retrieved question: {question}")
        print(f"Response: {response}")
        return response
    else:
        question, retrieved_response = get_retrieval_response(preprocessed_query)
        generated_response = generate_response(query, context=retrieved_response)
        print(f"Response: {generated_response}")
        return generated_response
    
# Example usages
while True:
    query = input("Enter your medical question (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    response = get_response(query)
    #print(f"Final response: {response}\n")
