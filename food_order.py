import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up Streamlit app
st.set_page_config(page_title="Food Order AI Assistant", page_icon="🍔")
st.title("Food Order AI Assistant")

# LLM setup
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
if not groq_api_key:
    st.info("Please enter your Groq API Key to continue.")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt template for food order interaction
food_prompt_template = PromptTemplate(
    input_variables=["user_input", "menu"],
    template="""
    You are a virtual food assistant. Based on the user's input and the menu provided, recommend dishes, take orders, 
    and answer any food-related questions.

    Menu:
    {menu}

    User Input:
    {user_input}

    Your Response:
    """
)

chain = LLMChain(llm=llm, prompt=food_prompt_template)

# Sample menu
menu = """
1. Margherita Pizza - $10
2. Vegan Burger - $12
3. Caesar Salad - $8
4. Spaghetti Carbonara - $15
5. Chocolate Brownie - $5
6. Smoothie (Mango, Berry, Green) - $6
"""

# Display menu
st.subheader("Menu")
st.text(menu)

# Chat interface
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm your food assistant. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.text_input("Your Message", placeholder="e.g., I want to order a pizza and a smoothie.")

if st.button("Send"):
    if user_input:
        with st.spinner("Processing..."):
            # Append user input to messages
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Generate response
            response = chain.run({"user_input": user_input, "menu": menu})

            # Append response to messages
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    else:
        st.warning("Please enter a message.")

