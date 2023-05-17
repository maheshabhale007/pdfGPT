import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

def main():
    # print("Hello World")
    load_dotenv()

    st.set_page_config(page_title='pdfhGPT', page_icon='🤖', layout='centered', initial_sidebar_state='auto')

    st.header("pdfGPT")

    # upload a pdf
    
    pdfs = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=True)
    
    # save the uploaded file locally
    if len(pdfs) != 0:
        # print(pdfs[0].id)
        for pdf in pdfs:
            with open(os.path.join("data_pdfFiles", pdf.name),"wb") as f: 
                f.write(pdf.getbuffer())         
            st.success("Saved File")

    loaders = []
    for pdf in pdfs:
        pdf_path = os.path.join("data_pdfFiles", pdf.name)
        loaders.append(UnstructuredPDFLoader(pdf_path))
        # the below gives error
        # loaders = [UnstructuredPDFLoader(pdf) for pdf in pdfs]
        # print(loaders)

    # print(loaders)

    if (len(loaders) != 0):
        index = VectorstoreIndexCreator().from_loaders(loaders)
        print(index)
    
    if (len(loaders) != 0):
        # print("Hello World"
        query = st.text_input("Ask a question:")
        st.write(query)
        if query:
            st.write('Query logged!')
            # print(index.query(query))
            results = index.query(query)
            st.write(results)
    
    # extracting the text from the pdf

if __name__ == '__main__':
    main()