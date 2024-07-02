import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

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
    st.write(pdf.name)

    if pdf is not None:
        pdf_reader = PdfReader (pdf)

        text = ""
        for page in pdf_reader.pages:   
            text += page.extract_text()

        
        # Splits the text into readable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len 
        )
        chunks = text_splitter.split_text(text=text)

        # Creates embeddings for the chunks
        embeddings = OpenAIEmbeddings()

        VectorStore = FAISS.from_texts(chunks, embeddings)

if __name__ == "__main__":
    main()