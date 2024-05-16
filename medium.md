# From Zero to Hero: Deploying a Large Language Model Chatbot with Docker, Langchain and GPT

<figure class="image">
  <img src="images/title_image.png" alt="Ai bot image">
  <figcaption><center>AI bot representation, AI generated. Author: self.</center></figcaption>
</figure>

Image generate with AI.Large Language Models (LLMs) represent a revolutionary stride in artificial intelligence. These sophisticated models are designed to understand and generate human-like text based on the data they've been trained on.

The genesis of LLMs can be traced back to research in natural language processing (NLP), a branch of artificial intelligence focused on the interaction between computers and human language. Over the years, as computational power surged and researchers refined their techniques, the ability of machines to parse, understand, and generate text has dramatically improved.

The real turning point came with the introduction of models like GPT (Generative Pre-trained Transformer) by OpenAI, which showcased an unprecedented fluency in handling diverse language tasks. These models are "pre-trained" on a large corpus of text from the internet, books, and articles, allowing them to develop a broad understanding of human language. After this pre-training, they can be "fine-tuned" on specific types of text to perform various tasks such as translation, summarization, or even creative writing.

The implications of LLMs are vast, touching everything from how we interact with digital assistants, to transforming educational tools, and even reshaping the landscape of creative industries. As digital natives, understanding the origins, development, and capabilities of these models not only provides a glimpse into the future of technology but also opens up myriad possibilities for innovative applications in your studies and future careers.

With these description, it seems that deploying your first custom solution using LLM must be a really hard task. But in the recent times, it became as simple as running a single line in a command prompt. In this article we will develop from the ground up, using OpenAI API and langchain, a simple chatbot. In the end we will also touch on Local LLM and how to deploy the same app using one.

## _Requierements_
- _OpenAI API token_
- _Python >= 3.10_
- _langchain, langchain_core and langchain_openai_


## 1. Making a basic chat model

Understanding how data moves through a large language model (LLM) like OpenAI's GPT (Generative Pre-trained Transformer) is fascinating and involves several steps. Here's a simplified breakdown of the process from the moment the data is inputted (in this case, text) to when the user receives an answer. Let's explore this process step-by-step:
### 1. Tokenization

The journey begins when you input your text into the LLM. The first step the model undertakes is tokenization. Tokenization is the process of converting the input text into smaller pieces, known as tokens. These tokens are not just words but can include parts of words or punctuation. For instance, the text "Don't stop learning!" might be broken down into tokens like ["Don't", "stop", "learning", "!"]. This is crucial because the model processes these tokens, not the original text.
### 2. Embedding

Once the text is tokenized, each token is converted into a numerical format that the model can understand. Imagine each token is like a word on a map. The computer creates a special set of coordinates (like latitude and longitude) that represents each word. These coordinates are like a secret code that the computer can understand.

Think of it like a special kind of GPS system for words. Just like how GPS coordinates help your phone navigate to a specific location, these special coordinates (vectors) help the computer understand the meaning and context of each word.

These special coordinates are learned by the computer when it's trained on a massive amount of text data. This training helps the computer understand how words relate to each other, including their meanings, grammar, and syntax.

<figure>
  <p align=center><iframe src="https://giphy.com/embed/l6oxet59Gdv1dU8Fva" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe></p>
  <figcaption><center>A representation of the embedding. The difference between swimming and duathlon is approximately running, showing the semantic relation between them. Author: Self.</center></figcaption>
</figure>

### 3. Passing through the Transformer Layers
After converting words into special math problems (vectors), the model uses a powerful architecture named transformer to understand the relationships between these words.

**Layers**: The transformer architecture is made up of multiple layers, like a stack of Legos. Each layer has two main parts: the attention mechanism and the feed-forward network.

**Attention Mechanism**: Imagine you're trying to understand a sentence, like "The cat sat on the mat." The attention mechanism helps the model focus on the most important words in the sentence. For example, when trying to understand the word "sat", the model will pay more attention to "cat" (because it's closely related) and less attention to "the" (because it's not as important). This helps the model understand how words work together to convey meaning.

**Feed-Forward Networks**: Once the model has focused on the important words, it uses a special kind of math operation called a feed-forward network to transform the information. Think of it like a special filter that helps the model understand complex relationships between words. This filter is applied to each word in the sentence, but with slightly different settings for each layer, allowing the model to capture subtle nuances in language.

In short, the transformer architecture is like a series of filters that help the model understand language. Each layer has two parts: the attention mechanism, which helps the model focus on important words, and the feed-forward network, which transforms the information to capture complex relationships.

### 4. Layer-by-Layer Processing
The transformer processes the input data through its multiple layers, with each layer passing its outputs up to the next. As the data moves through each layer, the model adjusts its internal representations of the input, refining its understanding and translating it into more abstract representations. This layered processing allows the model to consider various aspects of language like meaning, context, and grammar.

### 5. Output Generation
After the model has processed the input text through all the transformer layers, it's time to generate the output. Think of it like the final step in a recipe, where all the ingredients come together to create the finished dish.

**Predicting the Next Token**: The model uses the processed information to predict the next word in the sequence. Imagine you're trying to complete a sentence, and you need to guess the next word. The model does the same thing, but with a special math trick.

**Probability Distribution**: The model calculates the probability of each possible next word, like a list of options with a percentage chance of being correct. This is called a probability distribution.
**Softmax Function**: To convert these probabilities into a single answer, the model uses a special math function called softmax. Think of it like a filter that takes the list of options and assigns a confidence score to each one. The option with the highest score is chosen as the next word in the sequence.

**Logits to Probabilities**: The softmax function takes the raw predictions (called logits) from the last layer and converts them into probabilities. It's like taking a bunch of numbers and turning them into a percentage score, so the model can make a confident prediction.

### 6. Decoding
The last step is decoding the output tokens back into human-readable text. The model continues to predict subsequent tokens until it generates a complete answer or reaches a maximum length. Decoding strategies like greedy decoding, beam search, or top-k sampling may be used to enhance the quality or creativity of the generated text.
### 7. Returning the Answer
The decoded text is then presented as the answer to the user's input query. This entire process, from tokenization to providing an answer, typically happens in a matter of seconds, showcasing the power and efficiency of modern neural network models.
## Common Challenges and Limitations of LLMs
While Large Language Models (LLMs) have revolutionized the field of natural language processing, they are not without their challenges and limitations. Understanding these limitations is crucial for developers, researchers, and users alike.

### Bias and Stereotyping
LLMs are only as good as the data they're trained on. If the training data contains biases, the model will likely learn and perpetuate those biases. This can lead to stereotyping, discrimination, and unfair treatment of certain groups. For example, if a model is trained on text data that contains racial or gender biases, it may generate responses that perpetuate those biases.

### Overfitting and Underfitting
LLMs can suffer from overfitting, where the model becomes too specialized to the training data and fails to generalize well to new, unseen data. On the other hand, underfitting occurs when the model is too simple and fails to capture the underlying patterns in the data. Both scenarios can lead to poor performance and inaccurate results.

### Lack of Transparency and Explainability
LLMs are often seen as black boxes, making it difficult to understand why they generate certain responses. This lack of transparency and explainability can make it challenging to identify and address biases, errors, or inconsistencies in the model's behavior.

### Job Displacement
The automation of tasks using LLMs may lead to job displacement, particularly in industries where tasks are repetitive or can be easily automated.

As LLMs become more autonomous, it's essential to establish accountability mechanisms to ensure that their decisions can be explained and justified.
By acknowledging these challenges and limitations, we can work towards developing more responsible and ethical LLMs that benefit society as a whole.

## To the code
```python
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
```
## 1. Creating the model and the prompt.
```python
model = ChatOpenAI(model="gpt-4-turbo")
system_prompt = """You are a helpful assistant. You will recive the input from the user, if you don't know answer you don't know."""
prompt = ChatPromptTemplate.from_messages([
("system", system_prompt),
("human", "{input}")
])
```
In the realm of natural language processing, language models have emerged as a crucial component in generating human-like responses. In the context of Langchain's `ChatOpenAI`, the model itself is responsible for generating responses, with tokenization being a unique aspect of each Large Language Model (LLM).

Businesses and organizations have two primary options when it comes to leveraging language models: open-source and closed-source models. Open-source models, such as GPT-2 from OpenAI and Llama from Meta, offer free access to their weights, albeit with certain licensing restrictions. On the other hand, closed-source models, like GPT-4 from OpenAI and the Claude family of models from Anthropic, operate on cloud services, with their weights not publicly available.

The advantages of open-source models lie in their ability to be run locally, ensuring better privacy and data security. This is particularly important for applications that require offline access to information encoded in the model's weights. However, the computational expense of running large models like Mistras MOE 8x22b and Llama 70B can be a significant drawback, requiring expensive and specialized GPU hardware.

Fortunately, both open-source and closed-source models can be fine-tuned for specific tasks using tools like LoRA and QLoRA. This is particularly valuable for businesses with extensive datasets, as fine-tuning can significantly improve the model's accuracy for business-specific tasks. In a future article, we will delve deeper into the art of fine-tuning language models, exploring the vast potential of these powerful tools.

### 1.1. The Prompt-Model Interaction: Unlocking the Power of Language Generation

The intricate dance between the prompt and the language model is a fascinating aspect of natural language processing. In Langchain's ChatOpenAI, the prompt is carefully crafted using a `ChatPromptTemplate`, which provides a clear template for the model to follow. This template-based approach simplifies the model's task, enabling it to focus on generating relevant responses.

The system field plays a crucial role in guiding the model's behavior, while the user field captures the user's input. The combination of these two fields is then passed as a string to the Large Language Model (LLM). What's remarkable is that the model lacks an inherent understanding of the input as a question or prompt; instead, it simply continues generating text based on the input string. This process can be influenced by the definition of a stop token, which Langchain conveniently sets up when creating the model. However, if all stop tokens are cleared, the model will continue generating text until it reaches its context limit, which can be as high as 32,000 tokens for some models. This highlights the importance of consistency in using the same stop token when training a model, as advised by developers.

### 1.2. Orchestrating the Language Generation Pipeline
In Langchain's ChatOpenAI, the language generation process is orchestrated through a modular pipeline architecture. This pipeline is composed of several components, each responsible for a specific task in the language generation process.

The code snippet below illustrates the creation of a pipeline chain, which is a sequence of components that work together to generate a response to a user's query:
```python
chain = {"input": RunnablePassthrough()} | prompt | model | StrOutputParser()
output = chain.invoke("What is a LLM?")
```

Let's break down the components of this pipeline chain:

`RunnablePassthrough()`: This component serves as an input gateway, allowing the pipeline to receive user input.

`prompt`: This component is responsible for processing the user's input and generating a prompt that the language model can understand.

`model`: This is the language model itself, which generates a response based on the input prompt.

`StrOutputParser()`: This component takes the model's output and converts it into a human-readable string.

By invoking the pipeline chain with the input query "What is a LLM?", the system generates a response that answers the user's question. This modular architecture enables developers to easily customize and extend the language generation pipeline to meet specific business requirements.
## 2. Making a user interface
To make a simple user interface we can use the library `streamlit` that allow to develop web apps in plain python, making it much easier for prototyping our solution.
```python
st.title("LLM Assistant")
st.write("Ask me a question!")
input_question = st.text_input("Enter your question:")
if st.button("Ask"):
output = chain.invoke(input_question)
st.write("Answer:")
st.write(output)
```
This code creates a Streamlit app with a text input field where the user can enter a question. When the user clicks the "Ask" button, the app invokes the chain.invoke method with the user's input, and displays the output from the model.
You can run this code by saving it to a file (e.g. app.py) and running `streamlit run app.py` in your terminal. Then, open a web browser and navigate to http://localhost:8501 to access the app. That is all to run you first aplication, but if you want to deploy this app you might want to create a docker.
# 3. Creating a docker file
**Why Docker?**
Docker helps you deploy your Streamlit app in a consistent and reliable way. It's like packaging your app and all its dependencies into a single container that can be easily moved and run anywhere.
**Benefits:**
- Easy to deploy and manage
- Consistent performance across different environments
- Scalable and secure
- Easy to collaborate and share with others
Think of Docker like a shipping container for your app. You pack everything your app needs into the container, and then you can easily move it to any platform or environment, knowing it will work as expected.
Here is the Dockerfile and instructions to create a Docker container for the given code:
## 3.1 Creating a Dockerfile
The docker file is the base of our app, create a file and named `Dockerfile`, without any extension and place the code below in it. If you want a deeper dive on what this code is doing I adivise you to take a few classes in Docker and containization.
```
# Dockerfile
FROM python:3.11-bookworm
# Set the working directory to /app
WORKDIR /app
# Copy the requirements file
COPY requirements.txt .
# Install the dependencies
RUN pip install -r requirements.txt
# Copy the application code
COPY app.py .
# Set the environment variable for OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
# Expose the port
EXPOSE 8501
# Run the command to start the Streamlit app
CMD ["streamlit", "run", "app.py"]
```
Also create a `requirements.txt` for our python libraries.
```
# requirements.txt
streamlit
langchain_openai
openai
langchain_core
```
## 3.2 Building and Running the container
To build the Docker image, run the following command:
```
docker build -t my-llm-assistant .
```
To run the Docker container, run the following command:
```
docker run -p 8501:8501 -e OPENAI_API_KEY=<your_openai_api_key> my-llm-assistant
```
Replace `<your_openai_api_key>` with your actual OpenAI API key.
Once the container is running, you can access the Streamlit app by visiting `http://localhost:8501` in your web browser.

## Conclusion
In this article, we've explored the basics of Large Language Models (LLMs) and how to deploy a simple chatbot using OpenAI's API and Langchain. We've also discussed the challenges and limitations of LLMs, including bias, stereotyping, overfitting, and lack of transparency and explainability. By understanding these limitations, we can work towards developing more responsible and ethical LLMs that benefit society as a whole.

As we continue to push the boundaries of natural language processing, we're excited to explore new frontiers in AI research. In our next article, we'll delve into the world of Retrieval Augmented Generation (RAG), a powerful technique that combines the strengths of LLMs with the precision of retrieval-based models. Stay tuned to learn more about how RAG is revolutionizing the field of natural language processing!

In the meantime, be sure to check out the alternative versions of this project on my GitHub, including one that uses a Llama.cpp model. Happy coding!