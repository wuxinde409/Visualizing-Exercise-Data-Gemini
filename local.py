import streamlit as st
import google.generativeai as genai
genai.configure(api_key="AIzaSyCFWdJPCekccAXgWD2LFTtWMt3yB4sIVsE")
model= genai.GenerativeModel("models/gemini-2.0-flash")
st.set_page_config(page_title="local Gemini GUI", layout="centered")
st.title("Gemini  with context ")

if "conversation" not in st.session_state:
    st.session_state.conversation=[] #這邊能通過[-5:]這樣去表示只儲存最後的五個對話
    
user_input= st.text_input("You:", key="user_input")
if st.button("Send") and user_input.strip() !="":
    st.session_state.conversation.append({"role": "user", "parts": [user_input]})
    context = st.session_state.conversation
    try:
        response=model.generate_content(context)
        reply= response.text
    except Exception as e:
        reply =f"Error:{str(e)}"
    st.session_state.conversation.append({"role": "model", "parts": [reply]}) #將gemini回答加回array

#展示以往紀錄
st.markdown("chat history")
for message in st.session_state.conversation:
    if message["role"]=="user":
        # print(message)
        st.markdown(f"** You:{message['parts'][0]}")
    elif message["role"]=="model":
        st.markdown(f"** Gemini:{message['parts'][0]}")