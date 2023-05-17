import PyPDF2
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def main():
    load_dotenv()
    # print(os.getenv('OPENAI_API_KEY'))
    # print('Hello World')

    st.set_page_config(page_title='SprihGPT', page_icon='🤖', layout='centered', initial_sidebar_state='auto')

    st.header("SprihGPT")

    # upload a pdf
    pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    # extracting the text from the pdf
    if pdf is not None:
        pdf_reader = PyPDF2.PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # st.write(text)

        # splitting the sentences into chunks
        splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = splitter.split_text(text)

        # st.write(chunks)

        # creating the embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # taking input
        question = st.text_input("Ask a question:")
        if question:
            docs = knowledge_base.similarity_search(question)
            # st.write(docs)    

            # loading the model
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")

            kwargs = {
                'input_documnents': docs,
                'question': question,
            }

            response = chain.run(**kwargs)
            st.write(response)

if __name__ == '__main__':
    main()