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
import contextlib

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize NLTK tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the vectorizer and retrieval data
vectorizer = joblib.load('C:\\Users\\Eric\\Desktop\\chatbot\\models\\vectorizer.pkl')
questions, answers = joblib.load('C:\\Users\\Eric\\Desktop\\chatbot\\models\\qa_data.pkl')
X = vectorizer.fit_transform(questions)

# Load the generative model
model_path = 'C:\\Users\\Eric\\Desktop\\chatbot\\models\\generative_model.pth'
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

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
        outputs = model.generate(inputs, max_length=5000, num_return_sequences=1, no_repeat_ngram_size=2)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

def ask_for_clarification(query):
    print(f"I'm not sure I fully understand. Can you provide more context about '{query}'?")
    return input("Your clarification (or press Enter to skip): ")

""" def get_response(query):
    preprocessed_query = preprocess_query(query)
    retrieval_confidence = get_retrieval_confidence(preprocessed_query)

    if retrieval_confidence > 0.8:
        question, response = get_retrieval_response(preprocessed_query)
        print(f"Response: {response}")
        return response
    else:
        question, retrieved_response = get_retrieval_response(preprocessed_query)
        generated_response = generate_response(query, context=retrieved_response)
        print(f"Response: {generated_response}")
        return generated_response """

def get_response(query):
    preprocessed_query = preprocess_query(query)
    retrieval_confidence = get_retrieval_confidence(preprocessed_query)

    if retrieval_confidence > 0.8:
        question, response = get_retrieval_response(preprocessed_query)
        return response
    else:
        question, retrieved_response = get_retrieval_response(preprocessed_query)
        generated_response = generate_response(query, context=retrieved_response)
        return generated_response

"""     
# Example usages
while True:
    query = input("Enter your medical question (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    response = get_response(query)
    # Optionally, you can print the final response
    # print(f"Final response: {response}\n") """
