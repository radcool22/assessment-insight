import os 
import streamlit as st
from dotenv import load_dotenv
import pickle 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

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
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    # Creates embeddings for the chunks
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # loader = TextLoader("/Users/kabirgupta/Desktop/Python/assessment-insight/requirements.txt")
    # documents = loader.load()
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # docs = text_splitter.split_documents(documents)
    # db = FAISS.from_documents(docs, embeddings)
    # print(db.index.ntotal)

    # Lets the user upload a PDF report
    pdf = st.file_uploader("Upload your report card here", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

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
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        try:
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        except TimeoutError as e:
            st.error("Timeout error occurred while creating VectorStore. Please try again later.")
            
        store_name = pdf.name[:-4]
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)

if __name__ == "__main__":
    main()