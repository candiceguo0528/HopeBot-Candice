import streamlit as st
from streamlit_chat import message
import os
import time
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init
import base64

# --------------------------------------------------------------------------------------------------------------------------logic2END
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import streamlit as st
import openai
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


def text_to_speech(text):
    try:
        response = openai.audio.speech.create(model="tts-1", voice="nova", input=text)
        audio_path = "response_audio.mp3"
        with open(audio_path, "wb") as f:
            response.stream_to_file(audio_path)
        return audio_path
    except Exception as e:
        st.error(f"Text-to-speech conversion failed: {e}")
        return None

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    b64_audio = base64.b64encode(data).decode("utf-8")
    st.markdown(f"""<audio autoplay><source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3"></audio>""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "This is HopeBot, your mental health assistant. How can I assist you today? 😊"}]

# Display chat history
st.title("HopeBot: Your Mental Health Assistant 🤖")

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "🤗"):
        st.markdown(f"<p style='font-size: 24px; margin: 0;'>{message['content']}</p>", unsafe_allow_html=True)

# Chat input widget (text & audio)
with bottom():
    user_input = chat_input_widget()

# Handle user input
if user_input:
    if "text" in user_input:
        user_message = user_input['text']

    elif "audioFile" in user_input:
        audio_data = user_input["audioFile"]

        # 检查数据类型
        if isinstance(audio_data, list):
            if isinstance(audio_data[0], int):
                audio_data = bytes(audio_data)
            else:
                audio_data = b"".join(audio_data)
        elif isinstance(audio_data, bytes):
            pass
        else:
            st.error("未知的音频数据格式！")
            audio_data = None

        if audio_data:
            audio_path = "C:/Users/Bolin/Desktop/Hope/temp_audio.mp3"
            with open(audio_path, "wb") as f:
                f.write(bytes(audio_data))

            user_message = None  # 先初始化，确保不会出现未定义问题
            # 语音转文本（确保正确关闭文件）
            try:
                response = {}
                with open(audio_path, "rb") as audio_file:
                    msg = openai.audio.transcriptions.create(
                        model="whisper-1", response_format="text",
                        file=audio_file
                    )
                    response['text'] = msg
                    print(response)
                    if 'text' in response:
                        user_message = response['text']
                    else:
                        st.warning("未能从语音转录中获取文本。")
            except Exception as e:
                st.error(f"语音识别失败: {e}")
            finally:
                # 确保文件句柄关闭后再删除文件
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except PermissionError:
                        st.warning("无法立即删除音频文件，稍后会尝试自动清理。")

        else:
            user_message = None

    else:
        user_message = None

    # 处理用户消息
    if user_message:
        st.session_state.messages.append({"role": "user", "content": user_message})
        with st.chat_message("user", avatar="🤗"):
            st.markdown(f"<p style='font-size: 24px; margin: 0;'>{user_message}</p>", unsafe_allow_html=True)

        # 生成 HopeBot 的回复
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking 🤔..."):
                final_response = get_assistant_response(st.session_state.messages)

            st.markdown(f"<p style='font-size: 24px; margin: 0;'>{final_response}</p>", unsafe_allow_html=True)
            
            audio_file = text_to_speech(final_response)
            if audio_file:
                autoplay_audio(audio_file)
                # 确保音频播放完毕再删除
                try:
                    os.remove(audio_file)
                except PermissionError:
                    st.warning("无法立即删除生成的音频文件，稍后会尝试自动清理。")

            st.session_state.messages.append({"role": "assistant", "content": final_response})
