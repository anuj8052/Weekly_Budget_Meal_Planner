import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up Streamlit app
st.set_page_config(page_title="Budget-Friendly Meal Planner", page_icon="🥗")
st.title("Budget-Friendly Meal Planner")

# LLM setup
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
if not groq_api_key:
    st.info("Please enter your Groq API Key to continue.")
    st.stop()

llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Prompt template for meal planning
meal_prompt_template = PromptTemplate(
    input_variables=["preferences", "budget", "meals_per_week", "country", "city"],
    template="""
    You are an AI assistant specializing in meal planning. Based on the user's dietary preferences, weekly budget, 
    number of meals needed, country, and city, create a budget-friendly weekly meal plan. Ensure the plan meets nutritional needs, 
    includes a grocery list, and provides tips for reducing food waste. Consider the specific region, cultural preferences, and food availability 
    in {country}, {state}. Avoid suggesting ingredients or dishes that are culturally inappropriate or unavailable in the specified region.

    User Preferences:
    {preferences}

    Weekly Budget: ${budget}

    Meals Per Week: {meals_per_week}

    Country: {country}

    City: {state}

    Your Response:
    """
)

chain = LLMChain(llm=llm, prompt=meal_prompt_template)

# Input fields
st.sidebar.header("Your Preferences")
preferences = st.sidebar.text_area("Dietary Preferences", placeholder="e.g., vegetarian, high protein, low carb")
budget = st.sidebar.number_input("Weekly Budget ($)", min_value=10, step=5)
meals_per_week = st.sidebar.number_input("Meals Per Week", min_value=1, step=1)
country = st.sidebar.text_input("Country", placeholder="e.g., USA, India")
city = st.sidebar.text_input("State", placeholder="e.g., UP, MP")

# Check if inputs are provided
if not preferences:
    st.info("Please enter your dietary preferences in the sidebar.")
    st.stop()

if budget <= 0:
    st.warning("Please enter a valid weekly budget.")
    st.stop()

if meals_per_week <= 0:
    st.warning("Please enter the number of meals per week.")
    st.stop()

if not country:
    st.warning("Please enter your country.")
    st.stop()

if not city:
    st.warning("Please enter your city.")
    st.stop()

# Generate meal plan
if st.button("Generate Meal Plan"):
    with st.spinner("Generating your meal plan..."):
        response = chain.run({
            "preferences": preferences,
            "budget": budget,
            "meals_per_week": meals_per_week,
            "country": country,
            "city": city
        })

        # Display response
        st.subheader("Your Meal Plan")
        st.markdown(response)

# Footer
st.markdown("---")
st.caption("Powered by AI | Groq API & Gemma Model")
