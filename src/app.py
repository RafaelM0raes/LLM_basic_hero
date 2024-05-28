import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
model = ChatOpenAI(model="gpt-4-turbo", )
system_prompt = """You are a helpful assistant. You will recive the input from the user, if you don't know answer you don't know."""
prompt = ChatPromptTemplate.from_messages([
("system", system_prompt),
("human", "{input}")
])
chain = {"input": RunnablePassthrough()} | prompt | model | StrOutputParser()
st.title("LLM Assistant")
st.write("Ask me a question!")
input_question = st.text_input("Enter your question:")
if st.button("Ask"):
output = chain.invoke(input_question)
st.write("Answer:")
st.write(output)