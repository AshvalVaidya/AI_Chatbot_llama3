import os
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

# Function to interact with Groq AI
def chat_groq(messages):
    load_dotenv()
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    response_content = ''
    stream = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=1024,
        temperature=1.3,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            response_content += chunk.choices[0].delta.content
    return response_content

# Main function to control the app
def main():
    st.set_page_config(page_title='Simple Chatbot')

    st.title("AI Chatbot using llama3 LLM Model")
    with st.expander("Instructions"):
        st.write("1. Enter a question in the text area.")
        st.write("2. Submit the question to interact with the AI chatbot.")
        st.write("3. Recent chat history is displayed for reference.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = ""

    def submit_chat():
        user_message = st.session_state.user_input
        if user_message:
            prompt = f'''
            Answer the user's question based on the latest input provided in the chat history. Ignore
            previous inputs unless they are directly related to the latest question.

            Chat History: {st.session_state.chat_history}

            Latest Question: {user_message}
            '''

            messages = [{'role': 'system', 'content': 'You are a very helpful assistant'}]
            messages.append({'role': 'user', 'content': prompt})

            try:
                ai_response = chat_groq(messages)
            except Exception as e:
                st.error(f"Error occurred during chat_groq execution: {str(e)}")
                ai_response = "An error occurred while fetching the response. Please try again."

            # Display the current output prompt
            st.session_state.current_prompt = ai_response

            # Update chat history
            st.session_state.chat_history.append({'role': 'user', 'content': user_message})
            st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})

            # Clear the input field
            st.session_state.user_input = ""

    st.text_area("Enter your question:", key="user_input")
    st.button('Submit', on_click=submit_chat)

    # Display the current output prompt if available
    if st.session_state.current_prompt:
        st.write(st.session_state.current_prompt)

    # Display the last 4 messages in an expander
    with st.expander("Recent Chat History"):
        recent_history = st.session_state.chat_history[-8:][::-1]
        reversed_history = []
        for i in range(0, len(recent_history), 2):
            if i + 1 < len(recent_history):
                reversed_history.extend([recent_history[i + 1], recent_history[i]])
            else:
                reversed_history.append(recent_history[i])
        for chat in reversed_history:
            st.write(f"{chat['role'].capitalize()}: {chat['content']}")

if __name__ == "__main__":
    main()
