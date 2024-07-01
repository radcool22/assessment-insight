import streamlit as st
from PyPDF2 import PdfReader

with st.sidebar:
    # The sidebar with additional information about the website
    st.title("Chat with Report Cards")
    st.markdown("""
    ## About
    Hi! I am a student studying in Woodstock School 
    . This app is powered by Streamlit, OpenAI, and Langchain.
    """)

    st.write("Made by Kabir Gupta")

def main():
    # Prints the name of the product
    st.header("Assessment Insight")

    # Lets the user upload a PDF report
    pdf = st.file_uploader("Upload your report card here", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:   
            text += page.extract_text()

        st.write(text)

if __name__ == "__main__":
    main()