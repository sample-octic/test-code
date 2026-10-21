import os
import streamlit as st
from crewai import Agent, Crew, Process, Task

# Page Layout Configuration
st.set_page_config(
    page_title="CrewAI & Gemini Multi-Agent App", page_icon="🤖", layout="wide"
)

st.title("🤖 CrewAI Multi-Agent System with Gemini")
st.write(
    "Enter your Gemini API key and a topic below to let a **Research Analyst**"
    " and a **Content Writer** collaborate on your request."
)

# Sidebar for API Key & Model Settings
with st.sidebar:
  st.header("Configuration")
  api_key_input = st.text_input(
      "Gemini API Key",
      type="password",
      value=os.getenv("GOOGLE_API_KEY", ""),
      help="Get your key from Google AI Studio",
  )
  model_choice = st.selectbox(
      "Gemini Model", ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.5-flash"]
  )
  st.markdown("---")
  st.markdown("### How it works:")
  st.markdown("1. **Research Analyst**: Gathers raw insights.")
  st.markdown("2. **Content Writer**: Crafts the final markdown report.")

# Main Topic Input
topic = st.text_input(
    "What topic would you like the agents to work on?",
    placeholder="e.g., The future impact of Quantum Computing on Cybersecurity",
)

if st.button("🚀 Run Multi-Agent Crew"):
  if not api_key_input:
    st.error("Please enter a valid Gemini API Key in the sidebar.")
  elif not topic:
    st.warning("Please enter a topic for the agents to analyze.")
  else:
    with st.spinner(
        "Agents are initializing and collaborating... Please wait."
    ):
      try:
        # Set environment variable for Gemini
        os.environ["GOOGLE_API_KEY"] = api_key_input

        # Initialize Gemini LLM via LangChain wrapper
        # Configure Gemini for CrewAI
        llm = f"gemini/{model_choice}"
        # 1. Define Agents
        researcher = Agent(
            role="Senior Research Analyst",
            goal=(
                "Uncover cutting-edge facts, trends, and detailed insights on"
                f" {topic}"
            ),
            backstory=(
                "You are an expert researcher with a sharp eye for detail,"
                " able to pull together comprehensive information quickly."
            ),
            verbose=True,
            llm=llm,
        )

        writer = Agent(
            role="Technical Content Writer",
            goal=(
                "Transform raw research notes into a polished, engaging, and"
                " well-structured report."
            ),
            backstory=(
                "You are a professional writer who excels at turning complex"
                " technical or analytical data into clear, compelling narratives."
            ),
            verbose=True,
            llm=llm,
        )

        # 2. Define Tasks
        research_task = Task(
            description=(
                f"Conduct thorough research and compile key insights on: {topic}"
            ),
            expected_output=(
                "A comprehensive bullet-point list of key facts, trends, and"
                " technical insights."
            ),
            agent=researcher,
        )

        writing_task = Task(
            description=(
                "Take the research provided and write a professional, cohesive"
                " article/report summarizing the findings."
            ),
            expected_output=(
                "A well-structured markdown report with an introduction,"
                " detailed body sections, and a conclusion."
            ),
            agent=writer,
        )

        # 3. Form the Crew (Sequential workflow)
        crew = Crew(
            agents=[researcher, writer],
            tasks=[research_task, writing_task],
            process=Process.sequential,
            verbose=True,
        )

        # Execute Crew
        result = crew.kickoff()

        st.success("Crew execution completed successfully!")
        st.markdown("### 📄 Final Agent Report")
        st.markdown(str(result))

      except Exception as e:
        st.error(f"An error occurred during execution: {e}")