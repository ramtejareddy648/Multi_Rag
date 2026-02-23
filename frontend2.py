import streamlit as st
from backend2 import workflow,reterive_all_threads,delete_thread,MultiRag
import uuid
from langchain_core.messages import HumanMessage,AIMessage

from streamlit_mic_recorder import mic_recorder
import requests
import base64
import os
from dotenv import load_dotenv
load_dotenv()


SARVAM_API_KEY=os.getenv('SARVAM_API_KEY')

def speech_to_text(audio_bytes):
    """send audio to sarvam ai and get text"""
    url = "https://api.sarvam.ai/speech-to-text"
    
    files={
        'file':('audio.wav',audio_bytes,'audio/wav')
        
    }
    headers={
        'api-subscription-key':SARVAM_API_KEY
    }
    
    data = {
        'model': 'saaras:v3', 
        'mode': 'translate', 
        'language_code': 'te-IN'
    }

    try:
        response=requests.post(url,headers=headers,files=files,data=data)
        
        if response.status_code==200:
            return response.json().get('transcript','')
        else:
            st.error(f'sarvam Ai Error {response.text}')
            return None
    except Exception as e:
        st.error('failed to connect sarvam ai')
        return None

def text_to_speech(text):
    """convert text to speech"""
    url="https://api.sarvam.ai/text-to-speech"
    
    headers={
        'api-subscription-key':SARVAM_API_KEY,
        "Content-Type":'application/json'
    }
    
    payload = {
        'text': text,                    
        'model': 'bulbul:v3',
        'speaker': 'shubh',             
        'target_language_code': 'en-IN'
    }
    
    try:
        response=requests.post(url,headers=headers,json=payload)
        if response.status_code==200:
            return response.json()
        else:
            st.sidebar.error(f"TTS API Error {response.status_code}: {response.text}")
            return None
    except :
        return None




with st.sidebar:
    st.title('user session')
    
    user_name=st.text_input("Enter your Username to see your chats",value='Reddy')
    
    if st.button('Set Username'):
        if user_name:
            
            st.session_state['user_id'] = user_name.lower().strip()
            st.rerun()
    
    if 'user_id' not in st.session_state:
        st.warning("Please enter a username and click 'Set Username' to continue.")
        st.stop()
    
    else:
        st.success(f"Logged in as: **{st.session_state['user_id']}**")
    
    



with st.sidebar:
    st.write('## voice input')
    audio=mic_recorder(
        start_prompt="Start Recording",
        stop_prompt='Stop Recording',
        key='recorder'
    )


voice_text=""

if audio:
    if 'last_audio_id' not in st.session_state or st.session_state.last_audio_id!=audio['id']:
        st.session_state.last_audio_id=audio['id']
        
    
        voice_text=speech_to_text(audio['bytes'])
        
        if voice_text:
           st.session_state.voice_input=voice_text

    
    



def generate_thread_id():
    thread_id=uuid.uuid4()
    return thread_id



def add_thread_id(thread_id):
    if thread_id not in st.session_state['All_chat_threada']:
        st.session_state['All_chat_threada'].append(thread_id)


def reset_chat():
    thread_id=generate_thread_id()
    st.session_state['thread_id']=thread_id
    add_thread_id(thread_id)
    st.session_state['messages_history']=[]


def load_conversation(thread_id):
    try:
        state = workflow.get_state(config={'configurable': {'thread_id': thread_id}})
        return state.values.get('messages', [])
    except Exception as e:
        
        if "checkpoints" in str(e):
            return []
        raise e

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')


if 'messages_history' not in st.session_state:
    st.session_state['messages_history']=[]


if 'user_id' not in st.session_state:
    st.session_state['user_id'] = "guest_user"


if 'thread_id' not in st.session_state:
    st.session_state['thread_id']=generate_thread_id()


if 'user_id' in st.session_state:
    if 'All_chat_threada' not in st.session_state or st.session_state.get('last_user') != st.session_state['user_id']:
        st.session_state['All_chat_threada'] = reterive_all_threads(st.session_state['user_id'])
        st.session_state['last_user'] = st.session_state['user_id']


add_thread_id(st.session_state['thread_id'])

uploaded_files=st.sidebar.file_uploader(
    "upload fuiles or Images",
    type=['pdf','txt','png','jpg','jpeg'],
    accept_multiple_files=True
)


if st.sidebar.button('Process your files'):
    
    if uploaded_files:
        image_paths=[]
        file_paths=[]
        
        
        if not os.path.exists('uploads_doc'):
            os.makedirs('uploads_doc')
        
        
        with st.spinner('processing files this may take some time'):
            for uploaded_file in uploaded_files:
                file_path=os.path.join('uploads_doc',uploaded_file.name)
                with open(file_path,'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                
                if uploaded_file.name.lower().endswith(('pdf','txt')):
                    file_paths.append(file_path)
                else:
                    image_paths.append(file_path)
        rag_ingestor=MultiRag()
        rag_ingestor.build_enhanced_vector_database(
            image_paths,file_paths
        )
        st.sidebar.success(f"Successfully indexed {len(uploaded_files)} files!")
    else:
        st.sidebar.warning("Please upload files first.")
        

for thread_id in st.session_state['All_chat_threada'][::-1]:
    cols = st.sidebar.columns([0.8, 0.2])
    if cols[0].button(str(thread_id)[:8] + "...", key=f"select_{thread_id}"):
        st.session_state['thread_id']=thread_id
        messages=load_conversation(thread_id)
        
        
        temp_messages=[]
        for msg in messages:
            if isinstance(msg, HumanMessage) or (hasattr(msg, 'type') and msg.type == 'human'):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role':role,'content':msg.content})
        
        st.session_state['messages_history']=temp_messages
        st.rerun()
    
    if cols[1].button("🗑️", key=f"del_{thread_id}"):
        
        if delete_thread(thread_id):
            st.toast(f"Deleted chat {str(thread_id)[:8]}")
            
            st.session_state['All_chat_threada'] = reterive_all_threads(st.session_state['user_id'])
            if st.session_state.get('thread_id') == thread_id:
                st.session_state['thread_id'] = generate_thread_id()
                st.session_state['messages_history'] = []
               
            st.rerun()
        


for msg in st.session_state['messages_history']:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])


user_input=st.chat_input('Type Here')



if st.session_state.get('voice_input'):
    user_input=st.session_state.voice_input
    st.session_state.voice_input=None

if user_input:
    st.session_state['messages_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.markdown(user_input)
    
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"],
                         "user_id": st.session_state["user_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"],
            "user_id": st.session_state["user_id"]
        },
        "run_name": "chat_turn",
    }
    
    
    
    initial_input = {
        "question": user_input,
        "messages": [HumanMessage(content=user_input)],
        "docs": [],
        "good_docs": [],
        "web_docs": [],
        "refined_context_docs": []
    }
    
    with st.chat_message('assistant'):
        verdict_placeholder = st.empty()
        query_placeholder = st.empty()
        answer_placeholder = st.empty()
        
        full_answer = ""
        for update in workflow.stream(initial_input, config=CONFIG, stream_mode="updates"):
            for node_name, values in update.items():
                
                # if "verdict" in values:
                #     verdict_placeholder.markdown(f"**Verdict:** {values['verdict']}  \n**Reason:** {values.get('reason', 'N/A')}")
                
                # if "web_query" in values:
                #     query_placeholder.markdown(f"**Web Search Query:** `{values['web_query']}`")
                
                if "answer" in values:
                    full_answer = values["answer"]
                    answer_placeholder.markdown(full_answer)
        # if full_answer:
        #     with st.spinner('Generating audio response'):
        #         audio_response=text_to_speech(full_answer)
        #         if audio_response and "audios" in audio_response:
        #             try:
                    
        #                 audio_base64 = audio_response['audios'][0] 
        #                 audio_bytes = base64.b64decode(audio_base64)
        #                 filename = f"response_{uuid.uuid4().hex[:8]}.wav"
                        
        #                 with open(filename, "wb") as f:
        #                     f.write(audio_bytes)
        #                 st.success(f"✅ Audio saved locally as: `{filename}`")
        #                 st.audio(audio_bytes, format="audio/wav")
        #             except Exception as e:
        #                 st.error(f"Error saving audio: {e}")
                    
    st.session_state['messages_history'].append({'role':'assistant', 'content': full_answer})
