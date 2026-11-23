from dotenv import load_dotenv
load_dotenv()
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from rich import print
@tool
def get_text_length(text:str)->int:
    """return the number of character in the given text"""
    return(len(text))
tools ={"get_text_length":get_text_length}
llm = llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    request_timeout=60,
    max_retries=5
)

llm_with_tool = llm.bind_tools([get_text_length])
message =[]
prompt =input("you:")
query = HumanMessage(prompt)
message.append(query)
result = llm_with_tool.invoke(message)
message.append(result)
if result.tool_calls:
    print(result.tool_calls[0])
    tool_name =result.tool_calls[0]["name"]
    tool_message = tools[tool_name].invoke(result.tool_calls[0])
    message.append(tool_message)
result =llm_with_tool.invoke(message)
print(result.content)