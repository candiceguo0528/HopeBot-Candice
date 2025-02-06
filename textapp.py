import streamlit as st
# Display chat history
st.set_page_config(page_title="HopeBot", layout="wide")
import asyncio
import threading
import pyttsx3
from streamlit_chat import message
import os
import time
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
import base64
import openai
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
import chardet
import pysqlite3 as sqlite3
import sys
from streamlit_chat_widget import chat_input_widget
from streamlit_extras.bottom_container import bottom

st.set_page_config(page_title="HopeBot: Your Mental Health Assistant", layout="wide")
sys.modules["sqlite3"] = sqlite3
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Function to initialize resources
@st.cache_resource
def initialize_resources():
    # Chat model
    chat = ChatOpenAI(
        model="gpt-4o",
        temperature=0.4,
    )

    # Detect file encoding
    with open(r'cleaned_data.txt', 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']

    # Embedding model
    embed_model = OpenAIEmbeddings()

    # Vector stores
    vectorstore1 = Chroma(
        embedding_function=embed_model, 
        persist_directory="cleaned_data"
    )
    vectorstore2 = Chroma(
        embedding_function=embed_model, 
        persist_directory="mental_health"
    )
    vectorstore3 = Chroma(
        embedding_function=embed_model, 
        persist_directory="econ"
    )

    # Retrievers
    retriever1 = vectorstore1.as_retriever(k=2)
    retriever2 = vectorstore2.as_retriever(k=2)
    retriever3 = vectorstore3.as_retriever(k=2)

    # ChatPromptTemplate
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
             You are HopeBot, a professional psychotherapist specialising in Cognitive Behavioural Therapy. Your role is to focus on your clients' words and emotions, guiding them to reflect on their thoughts and behaviours through open-ended questions and guiding them through the PHQ-9 test. Always show empathy and understanding of their feelings and help them to recognise how their behaviour affects their emotions. Your responses should not be too long or presented in bullet point form, and all your responses should be spoken. You need to focus on listening, encourage clients to express themselves through short and precise language, and help them sort out and explore their emotions and thoughts. If a customer comes to you for advice, give up to 2 at a time. You need to provide helpful advice and assistance to users when they are experiencing extreme emotions, and start by adding encouraging sentences such as "You don't have to face this alone." 

    You must complete three tasks in turn:
    Task 1: Start by warmly greeting the client and creating a comfortable space for conversation. As a professional counselor, your goal is to listen attentively and engage in a natural flow of dialogue. As the conversation progresses, pay close attention to what the client shares. If they indicate that they have nothing else to share, or if the dialogue reaches about 20 exchanges, you must smoothly transition to introducing the PHQ-9 questionnaire and ask the user if they would like to take the PHQ-9 test. When doing this, acknowledge and validate what the client has shared so far, emphasizing how valuable their input has been.
    Task 2: After the user agrees to use the PHQ-9, ask each question in turn. Accurately categorise the user's answers as options A, B, C or D. If the user's answer is not precise enough, ambiguous or cannot be accurately categorised, you must ask the user to provide a clearer answer to ensure that the most accurate answer is collected, and you will need to ensure that the user completes all of the questions in turn. If the user answers A, they get 0 points; B, 1 point; C, 2 points; and D, 3 points. Track the score cumulatively without displaying it, and move to Task 3 after completing the test.
    Task 3: You must first tell the user of their answer distribution. In the format: Here’s how each answer was interpreted: Question 1: X (X point), etc. Then sum each question's mark up, and tell the user of their total score in number on the PHQ-9. In the format: You scored X points. If the user skipped questions, you need to mention how many questions the user skipped in your summary. And provide the appropriate depression severity results. Provide appropriate advice based on the results. If the depression is severe, give your advice and also encourage the user to seek professional help and provide them with a UK telephone helpline or email address (no more than 2 contacts). Be sure to make it clear that you are a virtual mental health assistant, not a doctor, and that whilst you will offer help, you are not a substitute for professional medical advice.
    At the end you will need to provide a brief summary of your conversation, including the confusion raised by the user in Task 1, as well as their PHQ-9 test results, and your corresponding recommendations. You need to ask the user if they have any further questions about the result and answer them.
    
    Please maintain the demeanour of a professional psychologist at all times and show empathy in your interactions. Please keep your responses concise and avoid giving long, repetitive answers.
    Here is some additional background information to help guide your responses:\n\n{context}
            """),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    # Create the LLM chain with the language model and the prompt
    document_chain = LLMChain(llm=chat, prompt=question_answering_prompt)

    # Return all initialized resources
    return chat, retriever1, retriever2, retriever3, question_answering_prompt, document_chain

# Initialize resources (runs once and caches results)
chat, retriever1, retriever2, retriever3, question_answering_prompt, document_chain = initialize_resources()

# Function to process input and return the chatbot's response
def get_assistant_response(messages):
    # Extract the user's last message (the latest user input)
    user_input = messages[-1]["content"]

    # Simulate chat history
    chat_history = ChatMessageHistory()
    for message in messages:
        chat_history.add_message(HumanMessage(content=message["content"]) if message["role"] == "user" else AIMessage(content=message["content"]))

    # Retrieve documents based on user input
    retriever_context = user_input  # Use user input as the query for document retrieval
    retrieved_docs1 = retriever1.get_relevant_documents(retriever_context)
    retrieved_docs2 = retriever2.get_relevant_documents(retriever_context)
    retrieved_docs3 = retriever3.get_relevant_documents(retriever_context)

    # Combine retrieved content into one context
    combined_context = "\n".join([doc.page_content for doc in retrieved_docs1 + retrieved_docs2 + retrieved_docs3])

    # Generate chatbot response with retrieved context
    response = document_chain.run(
        {
            "context": combined_context,  # Documents retrieved from retrievers
            "messages": chat_history.messages  # Conversation history
        }
    )

    # Return the assistant's response
    return response


# **异步 TTS 加速**
async def text_to_speech_async(text):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        response = await loop.run_in_executor(pool, lambda: openai.audio.speech.create(model="tts-1", voice="nova", input=text))
    return response.content  # 返回音频数据

# **同步调用封装**
def text_to_speech(text):
    return asyncio.run(text_to_speech_async(text))  # 调用 OpenAI 语音合成并返回数据

# **优化 autoplay_audio 直接在 streamlit 播放**
def autoplay_audio(audio_data):
    if audio_data:
        b64_audio = base64.b64encode(audio_data).decode("utf-8")
        st.audio(f"data:audio/mp3;base64,{b64_audio}", format="audio/mp3", autoplay=True)

# ------------------------------------------------------------------------------------------------------------------------------------------------logic2END
from streamlit_extras.stylable_container import stylable_container
 # Chat input widget (text & audio)
# with bottom():
with stylable_container(
    key="bottom_content",
    css_styles="""
        {
            position: fixed;
            bottom: 50px;
        }
        """,
):
    user_input = chat_input_widget()


st.title("HopeBot: Your Mental Health Assistant 🤖")   
# Float feature initialization
float_init()
with st.container(height=600):
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "This is HopeBot, your mental health assistant. How can I assist you today? 😊"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "🧐"):
            st.write(f"<p style='font-size: 24px; margin: 0;'>{message['content']}</p>", unsafe_allow_html=True)

    # Handle user input

    if user_input:
        user_message = user_input.get("text", None)

        # 处理语音输入
        if "audioFile" in user_input:
            audio_data = user_input["audioFile"]
            if isinstance(audio_data, list):
                audio_data = bytes(audio_data)
            elif not isinstance(audio_data, bytes):
                st.error("未知的音频数据格式！")
                audio_data = None

            if audio_data:
                with open("temp_audio.mp3", "wb") as f:
                    f.write(audio_data)
                
                try:
                    with open("temp_audio.mp3", "rb") as audio_file:
                        msg = openai.audio.transcriptions.create(
                            model="whisper-1", response_format="text",
                            file=audio_file
                        )
                        user_message = msg
                except Exception as e:
                    st.error(f"语音识别失败: {e}")

        if user_message:
            st.session_state.messages.append({"role": "user", "content": user_message})
            with st.chat_message("user", avatar="🧐"):
                st.markdown(f"<p style='font-size: 24px; margin: 0;'>{user_message}</p>", unsafe_allow_html=True)

            # 生成 HopeBot 的回复
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking 🤔..."):
                    final_response = get_assistant_response(st.session_state.messages)

                # **用 threading 提前加载语音**
                audio_data_holder = {"data": None}

                def generate_audio():
                    audio_data_holder["data"] = text_to_speech(final_response)

                thread = threading.Thread(target=generate_audio)
                thread.start()

                # **给 TTS 一点时间**
                time.sleep(1)  # 让 TTS 先跑一会儿

                # **先显示文本**
                st.markdown(f"<p style='font-size: 24px; margin: 0;'>{final_response}</p>", unsafe_allow_html=True)

                # **等待语音线程完成**
                thread.join()

                # **播放音频**
                autoplay_audio(audio_data_holder["data"])

                st.session_state.messages.append({"role": "assistant", "content": final_response})  
