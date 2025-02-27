import streamlit as st
import joblib

# Load the trained model and vectorizer
model = joblib.load("models/cyberbullying_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Streamlit app title
st.title("Cyberbullying Detection on Social Media Platforms ")

# Subtitle and instructions
st.markdown("""
Welcome to the Cyberbullying Detection App!  
Type or paste a social media post into the text box below and press **Enter** to analyze it for potential cyberbullying.
""")

# Add a placeholder for the result
result_placeholder = st.empty()

# User input
user_input = st.text_input("Enter a text (e.g., social media post):", key="text_input")

# JavaScript to trigger prediction on "Enter" keypress
st.markdown("""
<script>
const doc = window.parent.document;
const textInput = doc.querySelector("input[type='text']");
if (textInput) {
    textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent adding a newline in the textarea
            
            // Find the form submit button and trigger it
            const form = textInput.closest("form");
            if (form) {
                const buttons = form.querySelectorAll("button");
                const submitButton = Array.from(buttons).find(button => button.type === "submit");
                if (submitButton) {
                    submitButton.click(); // Simulate a click on the submit button
                }
            }
        }
    });
}
</script>
""", unsafe_allow_html=True)

# Prediction logic
if user_input.strip() != "":
    # Preprocess the input text
    input_transformed = vectorizer.transform([user_input])

    # Predict
    prediction = model.predict(input_transformed)[0]

    # Display result
    if prediction == 1:  # Assuming 1 means "cyberbullying"
        result_placeholder.error("⚠️ Cyberbullying Detected!")
    else:
        result_placeholder.success("✅ No Cyberbullying Detected.")
else:
    result_placeholder.warning("Please enter some text.")