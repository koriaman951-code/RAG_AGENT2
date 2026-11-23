from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnablePassthrough,RunnableParallel

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0.7,
    request_timeout=60,
    max_retries=5
)

parser = StrOutputParser()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a code generator"),
    ("human", "{topic}")
])

# Fixed string quotes around \n\n{code}
explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant who explain code in simple english language"),
    ("human", "Explain code in simple words:\n\n{code}")
])

# Fixed single pipes and added RunnableLambda bridge
seq = (
    code_prompt 
    | model 
    | parser  
)
seq2 = RunnableParallel(
    {"code": RunnablePassthrough(),
     "explanation": explain_prompt|model|parser
     
}
    )

chain = seq|seq2
result = chain.invoke({"topic": "please write a code of palidrom in python"})


# Print the string directly
print(result)                                                  
    
    


