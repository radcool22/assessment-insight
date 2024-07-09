import os 
import streamlit as st
from dotenv import load_dotenv
import pickle 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import openai
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback

with st.sidebar:
    # The sidebar with additional in formation about the website
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
    OPENAI_API_KEY = "sk-proj-u4TwE9KzXG7JWK2bT02dT3BlbkFJXJg8c3mncx32B1G8s90P"
    #OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl") and os.path.getsize(f"{store_name}.pkl") > 0:
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write("Embeddings loading from the Disk")
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            try:
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            except TimeoutError as e:
                st.error("Timeout error occurred while creating VectorStore. Please try again later.")
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        # Allowing the users to input a question
        query = st.text_input("Ask questions about your report card: ")

        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = openai(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
            st.write(response)

if __name__ == "__main__":
    main()