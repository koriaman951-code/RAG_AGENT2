import os
from dotenv import load_dotenv
from langchain_core.tools import tool
import requests
from rich import print
from tavily import TavilyClient
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

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
    print("DEBUG Weather:", data)

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


# 3. Initialize Gemini Model with an active model version
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    request_timeout=60,
    max_retries=5
)

# Map tools for lookup
tools = {
    "get_weather": get_weather,
    "get_news": get_news   
}

llm_with_tool = llm.bind_tools(list(tools.values()))
messages = []

print("City installed system (Gemini Powered)")
print("type exit to quit")

# 4. Interactive Chat Loop
while True:
    user_input = input("you: ")
    if user_input.lower() == "exit":
        break
    
    messages.append(HumanMessage(content=user_input))
    
    while True:
        result = llm_with_tool.invoke(messages)
        messages.append(result)
        
        if result.tool_calls:
            for tool_call in result.tool_calls:
                tool_name = tool_call["name"]
                
                confirm = input(f"agent wants to use {tool_name} approve (yes/no): ")
                if confirm.lower() == "no":
                    print("Tool execution cancelled by user.")
                    messages.append(ToolMessage(
                        content="Tool execution was denied by the user.",
                        tool_call_id=tool_call["id"]
                    ))
                    break
                
                tool_result = tools[tool_name].invoke(tool_call["args"])
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                ))
            continue
        else:
            print("AI Response:", result.content)
            break