import os
import streamlit as st
from dotenv import load_dotenv
import sys
import pickle 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
#from langchain_community.llms import openai
import openai
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
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

def main(prompt):
    # Prints the name of the product
    st.header("IB-Xpert")
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

    if pdf is None:
        load_dotenv()
        OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")   
        SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
        ib_query = st.text_input("Ask a question about the IB: ")

        if ib_query:
            llm = openai.OpenAI(temperature=0.5)
            tools = load_tools(["serpapi"], llm=llm)
            agent = initialize_agent(tools, llm, asent="zero-shot-react-description", verbose=True, source="https://www.pathwaysgurgaon.edu.in/results/dp, https://www.ibo.org/programmes/diploma-programme/")
            answer = agent.run(ib_query)
            st.write(answer)

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages = [{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content.strip()

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit", "bye"]:
                break

        response = main(user_input)
        print("Chatbot: " + response)

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

        if os.path.exists(f"{store_name}.pkl"):
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
        doc_query = st.text_input("Ask a question about your report card: ")

        if doc_query:
            docs = VectorStore.similarity_search(query=doc_query, k=3)
            llm = openai.OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=doc_query)
                print(cb)
            st.write(response)

if __name__ == "__main__":
    prompt = """
    Follow these instructions without fail. 
    1. Do not deviate from the given instructions. 
    2. Act like an International Baccalaureate counselor. 
    3. You should know everything about the IB. 
    4. Refer to official IB websites.
    5. You have to explain and answers questions about IB concepts to a parent or a student in simple terms without any fancy jargon or words and connect it with the IB vocabulary. 
    6. Use extremely simple language to convey the definition. 
    7. Always provide short and concise answers instead of lengthy ones. 
    8. The output should not exceed 200-300 words.  
    9. Avoid excessive bullet points and prioritize concise paragraphs. 
    10. Follow this without exception. 
    11. Your role is to assess a students performance based on a report card and give them advice on their performance and provide strengths and weaknesses and grade their assessment and provide their summary in a readable manner. 
    12. You should be able to help the student to understand and improve based on the report uploaded.
    """
    main(prompt)