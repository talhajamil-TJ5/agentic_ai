#Python code
#This is use to install openai agaent library

!pip install -Uq openai-agents


import nest_asyncio
nest_asyncio.apply()

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled

#Step 1: Setup for Api Keys

from google.colab import userdata
gemini_api_key = userdata.get("MY_NEW_GEMINI_KEY")

#Step 2: Client Setup for Connecting to Gemini

# Tracing disabled
set_tracing_disabled(disabled=True)

# 1. Which LLM Service?
external_client: AsyncOpenAI = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# 2. Which LLM Model?
llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

#Step 3 Running Agent Synchronously

math_agent: Agent = Agent(name="MathAgent",
                     instructions="You are a helpful math assistant.",
                     model=llm_model) # gemini-2.5 as agent brain - chat completions

result: Runner = Runner.run_sync(math_agent, "why learn math for AI Agents?")

print("\nCALLING AGENT\n")
print(result.final_output)


