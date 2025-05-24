from praisonaiagents import Agent, Task, PraisonAIAgents

vision_agent = Agent(
    name="ImageInterpreter",
    role="Visual Data Analyst",  # Fixed typo in "Anlyst"
    goal="To extract valuable insights and descriptions from images and video content",
    backstory="""As a specialist in visual content interpretation, 
                 you are adept at analyzing both static and dynamic images. 
                 Provide in-depth analysis of visual data.""",  # Fixed grammar and improved clarity
    llm="gpt-4o-mini",
    self_reflect=False
)

task_url = Task(
    name="landmark_description",
    description="Provide a detailed account of this well-known landmark and its architectural elements.",
    expected_output="A comprehensive description of the landmark's design and historical importance.",
    agent=vision_agent,
    images=["C:\Users\hardik anand\OneDrive\Desktop\Mutli-Modal AI Agent\pngtree-analysis-market-isolated-png-file-png-image_11462931.png"]  # Add a valid URL or path here
)

task_local_image = Task(
    name="object_identification",
    description="Analyze the objects within the image. Explain their arrangement and spatial relationships.",
    expected_output="A thorough breakdown of objects visible in the image and their relative positioning.",
    agent=vision_agent,
    images=[""]  # Add a valid local image path here
)

task_video = Task(
    name="video_analysis",  # Fixed lowercase for consistency
    description="""Watch the video and provide insights:
    1. A summary of main events occurring in the video.
    2. Important objects and people seen.""",
    expected_output="A detailed analysis of the video's key elements, including objects, people, and events.",
    agent=vision_agent,
    images=[""]  # Add a valid video file path or URL here
)

agents = PraisonAIAgents(
    agents=[vision_agent],  # Corrected from string to variable
    tasks=[task_url, task_local_image, task_video],  # Corrected from string to variable
    process="sequential",
    verbose=1
)

results = agents.start()
