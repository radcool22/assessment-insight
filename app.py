import os
import streamlit as st
from dotenv import load_dotenv
import sys
import pickle 
import openai as openai_original
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import openai
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback

with st.sidebar:
    st.title("Chat with Report Cards")
    st.markdown("""
    ## About
    Hi! I am a student studying in Woodstock School 
    . This app was creating by using Streamlit, OpenAI, and Langchain.
    """)
    st.write("Made by Kabir Gupta")

def main(prompt):
    st.header("🤖 IB-Xpert 📚📄")
    load_dotenv()
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")   
    SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

    pdf = st.file_uploader("Upload your report card here", type="pdf")
    query = st.text_input("Ask a question about IB or your report: ")
    
    def web_query():
        llm = openai.OpenAI(temperature=0.5)
        tools = load_tools(["serpapi"], llm=llm)
        if "pathways" in query.lower():
            agent = initialize_agent(tools, llm,
                                    asent="zero-shot-react-description",
                                    verbose=True,
                                    source="https://www.pathwaysgurgaon.edu.in/results/dp")
        elif "woodstock" in query.lower():
            agent = initialize_agent(tools, llm,
                                    asent="zero-shot-react-description",
                                    verbose=True,
                                    source="https://www.woodstockschool.in/")
        answer = agent.run(query)   
        st.write(answer)

    def ib_query():
        response = openai_original.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [{"role": "user", "content": query}]
        )

        answer = response.choices[0].message.content.strip()
        st.write(answer)

    def doc_query():
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
                st.write("Embeddings loading from the Disk")
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                try:
                    VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                except TimeoutError as e:
                    st.error("Timeout error occurred while creating VectorStore. Please try again later.")
                with open(f"{store_name}.pkl", "wb") as f:
                    st.write("Creating a new pickle file")
                    print(VectorStore)
                    pickle.dump(VectorStore, f)

            if query:
                docs = VectorStore.similarity_search(query=query, k=3)
                llm = openai.OpenAI(model_name="gpt-3.5-turbo")
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=query)
                    print(cb)
                st.write(response)

    if pdf:
        print("Calling doc_query()")
        doc_query()
    elif "woodstock" in query.lower() or "pathways" in query.lower():
        print("Calling web_query()")
        web_query()
    elif query:
        print("Calling ib_query()")
        ib_query()
    else:
        print("Loading....")

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