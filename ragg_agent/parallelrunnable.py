from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

# Use the exact model available to your current API key
model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    request_timeout=60,
    max_retries=5
)

parser = StrOutputParser()
topic = "machine learning"
short_prompt = ChatPromptTemplate.from_template("Explain {topic} in 1 short sentence.")
detailed_prompt = ChatPromptTemplate.from_template("Provide a detailed explanation of {topic}.")


chain = RunnableParallel({
    "short": RunnableLambda(lambda x:x["short"])|short_prompt | model | parser,
    "detailed": RunnableLambda(lambda x: x["detailed"]) |detailed_prompt | model | parser
})

result = chain.invoke({
    "short": {"topic":"machine learning"},
    "detailed":{"topic":"Deep Learning"}
})
print(result["short"])
print(result["detailed"])