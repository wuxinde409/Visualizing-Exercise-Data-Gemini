import os
import json
from datetime import datetime
import streamlit as st

# If you're using the official Google "generative ai" python SDK (common pattern in examples):
# pip install google-generativeai
# import google.generativeai as genai
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# This example uses a minimal wrapper so the code remains readable even if your SDK differs.
# Replace `call_gemini_chat` implementation with whatever your environment / SDK requires.

def call_gemini_chat(messages, model="gemini-1.5-pro"):
    """
    Small wrapper function for sending chat messages to Google Gemini.

    Replace this function body with the actual call for the SDK you have.
    Example for google.generativeai (pseudo):

        import google.generativeai as genai
        genai.configure(api_key=YOUR_KEY)
        resp = genai.chat.create(model=model, messages=messages)
        return resp.last

    For safety this function currently echoes a deterministic placeholder if API key is not set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Fallback behavior for local testing without a key
        user_text = messages[-1]["content"] if messages else ""
        return {"content": f"[No GEMINI_API_KEY found] Echo: {user_text}"}

    # --- Replace below with your real SDK call ---
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        resp = genai.chat.create(model=model, messages=[{"author": m["role"], "content": m["content"]} for m in messages])
        # The exact attribute to read depends on SDK version; adapt if needed
        return {"content": resp.last}
    except Exception as e:
        return {"content": f"[Gemini call failed: {e}]"}


# --------------------------- Memory (JSON) helpers ---------------------------

def default_fixed_filename():
    return "chat_memory.json"


def auto_filename(username: str = "anonymous"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_user = "".join(c for c in username if c.isalnum() or c in ("-","_")) or "user"
    return f"{safe_user}_{now}.json"


def load_memory_from_file(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []


def save_memory_to_file(path, messages):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# --------------------------- Streamlit UI ---------------------------

st.set_page_config(page_title="Gemini Chat with Persistent Memory", layout="wide")
st.title("Gemini Chat — Streamlit + Persistent JSON Memory")

# Sidebar: settings
st.sidebar.header("Settings")
model = st.sidebar.selectbox("Gemini model", ["gemini-1.5-pro", "gemini-pro", "gemini-pro-vision"], index=0)
use_fixed = st.sidebar.checkbox("Save to fixed filename (chat_memory.json)", value=True)
use_auto = st.sidebar.checkbox("Also save to auto-generated filename (username_date.json)", value=True)
username = st.sidebar.text_input("Username (for auto file)", value="user")

fixed_path = default_fixed_filename() if use_fixed else None

# If auto is enabled, generate a path to write at every run (so new session gets a timestamped file)
auto_path = auto_filename(username) if use_auto else None

st.sidebar.markdown("---")
st.sidebar.write("When both options are enabled, the app will save a copy to both files.")

# Conversation state stored in Streamlit session state for runtime, and written to disk for persistence.
if "messages" not in st.session_state:
    # Try to load from fixed file first (this mimics restoring the "latest" conversation automatically)
    if os.path.exists(default_fixed_filename()):
        st.session_state.messages = load_memory_from_file(default_fixed_filename())
        st.sidebar.success(f"Loaded memory from {default_fixed_filename()}")
    else:
        st.session_state.messages = []


# Show loaded history and allow clearing
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Conversation")
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Gemini:** {content}")

with col2:
    if st.button("Clear memory (session only)"):
        st.session_state.messages = []
        st.experimental_rerun()

st.markdown("---")

# Input box
user_input = st.text_input("Type your message and press Enter")
if user_input:
    # Append user message to runtime memory
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare messages for Gemini SDK: convert to expected format (role, content)
    sdk_messages = [(m["role"], m["content"]) for m in st.session_state.messages]

    # Call Gemini
    reply = call_gemini_chat([ {"role": r, "content": c} for r, c in sdk_messages ], model=model)
    text = reply.get("content", "[No reply]")

    # Append assistant reply
    st.session_state.messages.append({"role": "assistant", "content": text})

    # Persist to disk according to settings
    if fixed_path:
        save_memory_to_file(fixed_path, st.session_state.messages)
    if auto_path:
        save_memory_to_file(auto_path, st.session_state.messages)

    # Rerun so the UI displays the new messages
    st.experimental_rerun()

# Footer: instructions
st.markdown("---")
st.markdown("**Notes:**")
st.markdown("- Make sure you set environment variable `GEMINI_API_KEY` before running.\n- Install requirements: `pip install streamlit google-generativeai` (or your chosen SDK).\n- Run: `streamlit run streamlit_gemini_memory_app.py`")

st.caption("This app saves conversation history to JSON files so that closing and reopening the app can continue previous chats.")
