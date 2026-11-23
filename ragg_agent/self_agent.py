from dotenv import load_dotenv
import time

import os
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent
from langchain_core.rate_limiters import InMemoryRateLimiter

# 1. Define weather tool
@tool
def get_weather(city: str) -> str:
    """get current weather of city"""
    print(f"\n[INFO] Tool activated: get_weather (Fetching weather for {city})")
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric"}

    response = requests.get(url, params=params)
    data = response.json()
    

    if response.status_code != 200:
        return f"Error: {data.get('message', 'could not fetch')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    humidity = data["main"]["humidity"]

    return f"Weather in {city}: {temp}°C, {desc}, Humidity: {humidity}%"


# 2. Initialize Tavily client and news tool
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def get_news(city: str) -> str:
    """Get latest news about the city"""
    print(f"\n[INFO] Tool activated: get_news (Fetching news for {city})")
    
    query = f"latest news in {city}"
    response = tavily_client.search(
        query=query, search_depth="basic", max_results=5
    )

    results = response.get("results", [])
    if not results:
        return f"No recent news found for {city}.\n\n"

    formatted_news = f"Latest news for {city}:\n\n"
    for i, article in enumerate(results, 1):
        formatted_news += f"{i}. {article.get('title', 'No title')}\n"
        formatted_news += f"   {article.get('url', '')}\n\n"

    return formatted_news

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.02,   # slightly under 2 RPM to be safe
    check_every_n_seconds=0.1,
    max_bucket_size=3,          # room for decide + synthesize + 1 buffer
)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.7,
    timeout=60,
    max_retries=5,
    rate_limiter=rate_limiter,
)
agent = create_agent(
    llm,tools =[get_weather ,get_news],
    system_prompt= "you are a helpful city assistant."
)
print("city Agent| type exit to quit")
while True:
    user_input = input("you:")
    if user_input.lower() == "exit":
        break
    result = agent.invoke(
    {"messages": [HumanMessage(content=user_input)]}
    )
    print(result["messages"][-1])
    
    
    #switch to gemini rate limit was exusted .