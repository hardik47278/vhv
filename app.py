import streamlit as st
import cv2
import tempfile
from praisonaiagents import Agent, Task, PraisonAIAgents

OPENAI_API_KEY = "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"  # Add your OpenAI API key directly herest

# Initialize the Vision Agent
vision_agent = Agent(
    name="ImageInterpreter",
    role="Visual Data Analyst",
    goal="To extract valuable insights and descriptions from images and video content",
    backstory="""As a specialist in visual content interpretation, 
                 you are adept at analyzing both static and dynamic images. 
                 Provide in-depth analysis of visual data.""",
    llm="gpt-4o-mini",
    self_reflect=False
)

# Streamlit UI
st.title("Visual Data Analysis App")

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as temp_file:
        temp_file.write(uploaded_file.read())
        return temp_file.name

# Image Upload Section
uploaded_image = st.file_uploader("Upload an Image for Analysis", type=["jpg", "png", "jpeg"])
if uploaded_image:
    image_path = save_uploaded_file(uploaded_image)
    task_local_image = Task(
        name="object_identification",
        description="Analyze the objects within the image. Explain their arrangement and spatial relationships.",
        expected_output="A thorough breakdown of objects visible in the image and their relative positioning.",
        agent=vision_agent,
        images=[image_path]  # Using the uploaded image's path
    )

# Video Upload Section
uploaded_video = st.file_uploader("Upload a Video for Analysis", type=["mp4", "mov", "avi"])
if uploaded_video:
    video_path = save_uploaded_file(uploaded_video)
    task_video = Task(
        name="video_analysis",
        description="""Watch the video and provide insights:
        1. A summary of main events occurring in the video.
        2. Important objects and people seen.""",
        expected_output="A detailed analysis of the video's key elements, including objects, people, and events.",
        agent=vision_agent,
        images=[video_path]  # Using the uploaded video's path
    )

# Run Analysis Button
if st.button("Run Analysis"):
    tasks = []
    if uploaded_image:
        tasks.append(task_local_image)
    if uploaded_video:
        tasks.append(task_video)

    if tasks:
        agents = PraisonAIAgents(
            agents=[vision_agent],
            tasks=tasks,
            process="sequential",
            verbose=1
        )
        results = agents.start()
        st.write("### Analysis Results:")
        st.json(results)
    else:
        st.warning("Please upload an image or video before running the analysis.")
