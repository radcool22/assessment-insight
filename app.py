import os
import streamlit as st
from dotenv import load_dotenv
import openai as openai_original
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import openai
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent   
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks.manager import get_openai_callback

with st.sidebar:
    st.title("AI-Powered Insights on the IB and Assessment Reports")
    st.markdown("""
    ## About
    Hi! I am a student currently in Grade 10. 
    This app is a way for students and parents to easily 
    be able to understand the IB and their report cards.
    """)
    st.write("Made by Kabir Gupta")
    OPENAI_API_KEY = "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"

def main(prompt):
    st.header("📚 IB-Xpert 📄")
    load_dotenv()
    SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

    pdf = st.file_uploader("Upload your report card here", type="pdf")
    query = st.text_input("Ask a question about IB or your report: ")

    if OPENAI_API_KEY:
        def web_query():
            llm = openai.OpenAI(api_key=OPENAI_API_KEY)
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
            openai_instance = openai_original.OpenAI(api_key=OPENAI_API_KEY)
            response = openai_instance.chat.completions.create(
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

                # This line Splits the text into readable chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    separators="\n",
                    chunk_size = 1000,
                    chunk_overlap = 200,
                    length_function = len 
                )
                chunks = text_splitter.split_text(text)
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                vectorstore = FAISS.from_texts(chunks, embeddings)

                if query:
                    docs = vectorstore.similarity_search(query=query, k=3)
                    llm = openai.OpenAI(api_key=OPENAI_API_KEY)
                    chain = load_qa_chain(llm=llm, chain_type="stuff")
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=query)
                        print(cb)
                    st.write(response)

        if pdf:
            doc_query()
        elif "woodstock" in query.lower() or "pathways" in query.lower():
            web_query()
        elif query:
            ib_query()
        else:
            print("Loading....")
    else:
        st.info("Please add your OpenAI key to continue")
        st.stop()

if __name__ == "__main__":
    prompt = """
    Follow these instructions without fail. 
    1. Act like an International Baccalaureate counselor. 
    2. You should know everything about the IB. 
    3. Refer to official IB websites.
    4. You have to explain and answers questions about IB concepts to a parent or a student 
    in simple terms without any fancy jargon or words and connect it with the IB vocabulary. 
    5. Use extremely simple language to convey the definition. 
    6. Always provide short and concise answers instead of lengthy ones. 
    7. The output should not exceed 200-300 words.  
    8. Avoid excessive bullet points and prioritize concise paragraphs. 
    9. Your role is to assess a students performance based on a report card and give them 
    advice on their performance and provide strengths and weaknesses and grade their assessment and 
    provide their summary in a readable manner. 
    10. You should be able to help the student to understand and improve based on the report uploaded.
    """
    main(prompt)