import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.chains.question_answering import load_qa_chain
import openai
# Import Azure OpenAI
from langchain_openai import AzureOpenAI, AzureChatOpenAI
import os
from langchain_community.callbacks import get_openai_callback
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "<OpenAI_API_Key>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "<OpenAI_Endpoint>"
os.environ["OPENAI_API_VERSION"] = "2024-02-01"
 
# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App - LLama - 🦙')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
 
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by Arik Bidny ')
 
load_dotenv()
 
def main():
    st.header("Chat with PDF 💬 and Llama - 🦙 ")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # print(text)
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)


        embeddings = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-ada-002",
            )
        
        db = FAISS.from_texts(chunks, embedding=embeddings)
        print(db.index.ntotal)

        query = st.text_input("Ask questions about your PDF file:")
        st.write(query)

        if query:
            retriever = db.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            # print(docs)
            # print(len(docs))
            # docs = db.similarity_search(query=query, k=3)
            # print(docs[0].page_content)

            ######## WORKING WITH AZURE OPENAI #########
            # llm = AzureChatOpenAI(
            #     deployment_name="gpt-4-32k"
            # )

            # chain = load_qa_chain(llm=llm, chain_type="stuff")
            # with get_openai_callback() as cb:
            #     response = chain.run(input_documents=docs, question=query)
            #     print(cb)
            # st.write(response)

            ######## WORKING WITH LLama #########
            context = docs[0].page_content
            # print(context)

            ### LAMMA CONFIGURATION ###
            client =  OpenAI(base_url="http://localhost:1234/v1", api_key="")

            prompt = f"You are a helpful assistant to answer questions based on this context: {context}, the question is: {query}"
            print(prompt)

            completion = client.chat.completions.create(
                model="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0
            )

            print(completion.choices[0].message)
            st.write(completion.choices[0].message)
        # if os.path.exists(f"{store_name}.pkl"):
        #     with open(f"{store_name}.pkl", "rb") as f:
        #         VectorStore = pickle.load(f)
        #     # st.write('Embeddings Loaded from the Disk')s
        # else:
        #     # embeddings = OpenAIEmbeddings()
        #     embeddings = AzureOpenAIEmbeddings(
        #         azure_deployment="text-embedding-ada-002",
        #     )
        #     VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        #     with open(f"{store_name}.pkl", "wb") as f:
        #         pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        # query = st.text_input("Ask questions about your PDF file:")
        # st.write(query)
 
        # if query:
        #     docs = VectorStore.similarity_search(query=query, k=3)
 
        #     # llm = OpenAI()
        #     llm = AzureOpenAI(
        #         deployment_name="gpt-4"
        #     )
        #     chain = load_qa_chain(llm=llm, chain_type="stuff")
        #     with get_openai_callback() as cb:
        #         response = chain.run(input_documents=docs, question=query)
        #         print(cb)
        #     st.write(response)
 
if __name__ == '__main__':
    main()