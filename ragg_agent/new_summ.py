from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import TavilySearchResults

# Fixed class name (TavilySearchResults) and parameter (max_results)
search_tool = TavilySearchResults(max_results=5)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    request_timeout=60,
    max_retries=5
)

prompt = ChatPromptTemplate.from_template(
    """you are helpful ai assistant
    summarize the following into clear bullet points 
    {news}
    """
)

chain = prompt | llm | StrOutputParser()

# Fixed deprecated .run() to .invoke()
news_result = search_tool.invoke("Latest AI news of 2026")

# Fixed dictionary syntax: {"news": news_result}
result = chain.invoke({"news": news_result})

print(result)

print(search_tool.name)
print(search_tool.description)
print(search_tool.args)