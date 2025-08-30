import streamlit as st
from transformers import pipeline

# Load a code generation model (you can change the model if you want)
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="microsoft/CodeGPT-small-py")

generator = load_model()

# Streamlit UI
st.title("💻 AI Code Generator")
st.write("Generate Python code from your prompt using Hugging Face models.")

# Input from user
prompt = st.text_area("Enter your prompt:", placeholder="e.g., Write a function to reverse a string in Python")

if st.button("Generate Code"):
    if prompt.strip():
        with st.spinner("Generating code..."):
            output = generator(prompt, max_length=150, num_return_sequences=1, do_sample=True, temperature=0.7)
            code = output[0]['generated_text']
        st.subheader("Generated Code:")
        st.code(code, language="python")
    else:
        st.warning("Please enter a prompt.")
