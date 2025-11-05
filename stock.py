import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup
import requests
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Dict
from langchain.agents import AgentOutputParser
from langchain.chains import LLMChain
import re
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# ---------------------------
# Set up Gemini API key
# ---------------------------
# Set the API key as an environment variable
os.environ["GOOGLE_API_KEY"] = "GEMINI-API-KEY"

# ---------------------------
# Single Toggle for Dark Mode / Light Mode
# ---------------------------
# A checkbox to enable dark mode (black color combo) or light mode (white color combo)
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
if dark_mode:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #000000;
            color: #FFFFFF;
        }
        [data-testid="stSidebar"] {
            background-color: #222222;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #333333;
            color: #FFFFFF;
            border: none;
        }
        .stTextInput>div>div>input {
            background-color: #444444;
            color: #FFFFFF;
            border: 1px solid #555555;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #FFFFFF;
            color: #000000;
        }
        [data-testid="stSidebar"] {
            background-color: #F0F0F0;
            color: #000000;
        }
        .stButton>button {
            background-color: #E0E0E0;
            color: #000000;
            border: none;
        }
        .stTextInput>div>div>input {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #CCCCCC;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------
# Custom prompt template for the agent
# ---------------------------
class StockAnalysisPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += f"Action: {action.log}\nObservation: {observation}\n"
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        kwargs["thoughts"] = thoughts
        return self.template.format(**kwargs)

# ---------------------------
# Initialize Gemini model configuration
# ---------------------------
@st.cache_resource
def load_model():
    # Initialize the Gemini model with the updated model name
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # Updated to gemini-2.5-flash
        temperature=0,
        convert_system_message_to_human=True  # This helps with prompt formatting
    )
    return llm

# ---------------------------
# Custom tools for the agent
# ---------------------------
def get_stock_data(ticker: str) -> Dict:
    """Get basic stock information for a given ticker symbol."""
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "currentPrice": info.get("currentPrice"),
        "marketCap": info.get("marketCap"),
        "longName": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "website": info.get("website"),
        "longBusinessSummary": info.get("longBusinessSummary")
    }

def get_stock_news(ticker: str):
    """
    Fetches the latest news articles for a given stock ticker from Google News.
    
    :param ticker: Stock ticker symbol (e.g., "AAPL" for Apple)
    :return: A list of dictionaries containing news titles and URLs.
    """
    base_url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(base_url, headers=headers)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return []
    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.find_all("div", class_="SoaBEf")
    news_list = []
    for article in articles:
        title_tag = article.find("div", class_="n0jPhd ynAwRc MBeuO nDgy9d")
        if title_tag:
            title = title_tag.text.strip()
            link_tag = article.find("a")
            link = f"https://www.google.com{link_tag['href']}" if link_tag else None
            news_list.append({"title": title, "url": link})
    return news_list

def get_stock_financials(ticker: str) -> Dict:
    """Get key financial metrics for a given stock ticker symbol."""
    stock = yf.Ticker(ticker)
    try:
        financials = stock.financials
        return {
            "Revenue": financials.loc["Total Revenue"].to_dict() if "Total Revenue" in financials.index else {},
            "Net Income": financials.loc["Net Income"].to_dict() if "Net Income" in financials.index else {},
            "Operating Income": financials.loc["Operating Income"].to_dict() if "Operating Income" in financials.index else {}
        }
    except Exception as e:
        return {"error": f"Unable to fetch financial data: {e}"}

# ---------------------------
# Create the agent
# ---------------------------
def create_stock_agent(llm):
    tools = [
        Tool(
            name="get_stock_info",
            func=get_stock_data,
            description="Get basic information about a stock. Input should be a valid stock ticker symbol."
        ),
        Tool(
            name="get_stock_news",
            func=get_stock_news,
            description="Fetch recent news articles about a stock. Input should be a valid stock ticker symbol."
        ),
        Tool(
            name="get_stock_financials",
            func=get_stock_financials,
            description="Get key financial metrics for a stock. Input should be a valid stock ticker symbol."
        )
    ]
    
    # Updated prompt: instructs the agent to answer any questions related to the stock market.
    template = """You are a helpful AI stock research assistant specialized in answering any questions related to the stock market.
Your domain includes stock prices, company information, market news, financial data, trends, and overall market analysis.

You have access to the following tools:
{tools}

Use the following format when needed:
Question: the input question you must answer
Thought: you should always think about what to do next
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation:
{thoughts}

Question: {input}"""

    prompt = StockAnalysisPromptTemplate(
        template=template,
        tools=tools,
        input_variables=["input", "intermediate_steps"]
    )
    
    class StockAgentOutputParser(AgentOutputParser):
        def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
            if "Final Answer:" in llm_output:
                return AgentFinish(
                    return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                    log=llm_output,
                )
            
            regex = r"Action: (.*?)[\n]*Action Input: (.*)"
            match = re.search(regex, llm_output, re.DOTALL)
            
            if not match:
                # If no valid action is found, treat the entire output as the final answer.
                return AgentFinish(
                    return_values={"output": llm_output.strip()},
                    log=llm_output,
                )
                
            action = match.group(1).strip()
            action_input = match.group(2).strip()
            
            return AgentAction(tool=action, tool_input=action_input, log=llm_output)
    
    output_parser = StockAgentOutputParser()
    
    agent = LLMSingleActionAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=[tool.name for tool in tools]
    )
    
    memory = ConversationBufferMemory(memory_key="thoughts")
    
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

# ---------------------------
# Streamlit UI
# ---------------------------
st.title('AI Stock Research Assistant')
st.write('Ask any question related to the stock market—from stock prices and news to financial analysis and market trends!')

# Initialize session state with the agent if not already loaded
if 'agent' not in st.session_state:
    with st.spinner('Loading Gemini model...'):
        llm = load_model()
        st.session_state.agent = create_stock_agent(llm)

# User input
user_question = st.text_input('Ask me anything about the stock market:')

if user_question:
    with st.spinner('Analyzing...'):
        try:
            response = st.session_state.agent.run({"input": user_question})
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Powered by LangChain, Google Gemini, and Streamlit by Abhishek Gupta")
make better ui and remove dark theme make milky white combo chabot with font is fonts/batmfa__.ttf and for bot output font is fonts/Montserrat-Light.ttf and bot icon <i class="fa-solid fa-message-bot"></i>
