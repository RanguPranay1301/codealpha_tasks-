import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Sample FAQ dataset
faqs = [
    {"question": "What is your return policy?", "answer": "Our return policy allows returns within 30 days of purchase."},
    {"question": "How do I track my order?", "answer": "You can track your order using the tracking link sent to your email."},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship to most countries worldwide."},
    {"question": "How can I contact customer service?", "answer": "You can contact customer service via our support page or call us directly."},
    {"question": "What payment methods do you accept?", "answer": "We accept credit/debit cards, PayPal, and UPI."}
]

# Preprocessing function
stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Preprocess all FAQ questions
faq_questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess(q) for q in faq_questions]

# Response generation
def get_response(user_input):
    user_input_processed = preprocess(user_input)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([user_input_processed] + processed_questions)
    similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    max_index = similarity.argmax()

    if similarity[max_index] > 0.3:  # confidence threshold
        return faqs[max_index]["answer"]
    else:
        return "Sorry, I couldn't find a relevant answer. Please try rephrasing."

# Chat loop
def run_chatbot():
    print("🤖 FAQ Chatbot is online! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break
        response = get_response(user_input)
        print("Bot:", response)

# Run chatbot
if __name__ == "__main__":
    run_chatbot()
