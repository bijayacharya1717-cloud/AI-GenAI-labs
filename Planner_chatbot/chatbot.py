import os
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.tools.executors import SequentialToolExecutor

@tool
def get_weather(city: str) -> str:
    """Get the weather for a city.

    Args:
        city: Name of the city.

    Returns:
        Weather information for the city.
    """
    weather = {
        "kathmandu": "Sunny, 24°C",
        "pokhara": "Partly cloudy, 22°C",
        "london": "Cloudy, 15°C",
        "new york": "Sunny, 20°C",
    }
    return weather.get(city.lower(), f"Weather data not available for {city}.")


@tool 
def search_attractions(city: str) -> str:
    """Searches for the top 3 attractions of the given city.

    Args:
        city: Name of the city.

    Returns:
        Top 3 attractions for the city.
    """
    attractions = {
        "kathmandu": "1. Pashupatinath Temple, 2. Boudhanath Stupa, 3. Swayambhunath",
        "london": "1. Tower of London, 2. British Museum, 3. London Eye",
        "pokhara": "1. Phewa Lake, 2. World Peace Pagoda, 3. Davis Falls",
        "new york": "1. Central Park, 2. Statue of Liberty, 3. Empire State Building",
    }
    return attractions.get(city.lower(), f"No attractions found for {city}.")


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: A mathematical expression such as "50 * 3 + 100".

    Returns:
        The calculated result.
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Could not calculate the expression: {e}"

model = OpenAIModel(
    model_id="openai/gpt-oss-120b",
    client_args={
        "api_key": os.getenv("GROQ_API_KEY"),
        "base_url": "https://api.groq.com/openai/v1"
    }
)

agent = Agent(
    model=model,
    tools=[
        search_attractions,
        get_weather,
        calculator,
    ],
    tool_executor=SequentialToolExecutor(),
    system_prompt="You are a helpful travel planner assistant. When a user asks for an itinerary for a city, use the get_weather tool to check the weather, the search_attractions tool to find top attractions, and the calculator tool to calculate estimated trip/admission costs. Then, present a complete one-day itinerary including weather, attractions, and total estimated cost clearly."
   
)

def chat(message, history):
    response = agent(message)
    return str(response)


gr.ChatInterface(
    fn=chat,
    title="Strands Tool powered Chatbot",
    description="A chatbot powered by Strands + Groq + GPT-OSS-120B",
).launch()


