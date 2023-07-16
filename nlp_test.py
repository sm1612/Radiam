import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Download stopwords from NLTK
nltk.download('stopwords')

# Preprocess the text data
def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    
    # Join the filtered tokens back into a single string
    preprocessed_text = ' '.join(filtered_tokens)
    
    return preprocessed_text

# Define the training data
training_data = [
    ("I have a headache", "medical"),
    ("I broke my arm", "medical"),
    ("I love pizza", "non-medical"),
    ("I enjoy playing soccer", "non-medical")
]

# Preprocess the training data
preprocessed_training_data = [(preprocess_text(text), label) for text, label in training_data]

# Extract features from the training data using TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform([text for text, _ in preprocessed_training_data])
y_train = [label for _, label in preprocessed_training_data]

# Train the classifier
classifier = LinearSVC()
classifier.fit(X_train, y_train)

# Function to classify new text
def classify_text(text):
    preprocessed_text = preprocess_text(text)
    X_test = vectorizer.transform([preprocessed_text])
    prediction = classifier.predict(X_test)
    return prediction[0]

# Test the classifier
test_text = "I have a fever"
classification = classify_text(test_text)
print(f"Text: {test_text}")
print(f"Classification: {classification}")
