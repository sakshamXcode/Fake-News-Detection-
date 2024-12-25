import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Ensure necessary NLTK data is available
nltk.download('stopwords')

# Initialize the PorterStemmer and load the model/vectorizer
port_stem = PorterStemmer()

try:
    vector_form = pickle.load(open('vector.pkl', 'rb'))  # Load the vectorizer
    load_model = pickle.load(open('model.pkl', 'rb'))    # Load the pre-trained model
except FileNotFoundError:
    st.error("Model or vectorizer file not found.")
    st.stop()

# Text preprocessing function
def preprocess_text(content):
    # Remove non-alphabetic characters and lowercase the text
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()

    # Remove stopwords and apply stemming
    stop_words = set(stopwords.words('english'))
    content = [port_stem.stem(word) for word in content if word not in stop_words]
    
    # Join words back into a single string
    content = ' '.join(content)
    return content

# Prediction function
def predict_news(content):
    content = preprocess_text(content)  # Preprocess the content
    input_data = [content]  # Prepare the input data for prediction

    # Vectorize the input using the pre-loaded vectorizer
    vectorized_input = vector_form.transform(input_data)
    
    # Make the prediction
    prediction = load_model.predict(vectorized_input)
    return prediction

# Streamlit UI Setup
st.title("Fake News Classification App")
st.subheader("Enter News Content to Check its Reliability")

# User input section for news content
user_input = st.text_area("News Content", height=200)

# When the user clicks the 'Predict' button
if st.button("Predict"):
    if user_input.strip() == "":  # Check if input is empty
        st.warning("Please enter some text for analysis.")
    else:
        # Get prediction result
        prediction = predict_news(user_input)
        
        # Debugging: Print prediction in the terminal
        print(f"Prediction: {prediction[0]}")  # Log the prediction result

        # Display results based on prediction in Streamlit UI
        if prediction == [0]:  # Reliable news
            st.success("✅ This news is classified as **Reliable**.")
        elif prediction == [1]:  # Unreliable news
            st.warning("❌ This news is classified as **Unreliable**.")
