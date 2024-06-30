from typing import Union, List

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, Tool
from langchain_openai import chat_models
from callbacks import AgentsCallbackHandler

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""  # imp to give this as this  tells LLM what this do and is it useful ?
    text = text.strip().strip("\"'\n")
    return len(text)

class output_parser_func(ReActSingleInputOutputParser):
    def parse(self, text: str):
        if "Observation: " in text:
            action, observation = text.split("Observation: ")
            return super().parse(action)
        else:
            return super().parse(text)



def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for i in tools:
        if i.name == tool_name:
            return i
    raise ValueError(f"No Tool found with name as : {tool_name}")

if __name__ == '__main__':
    print("Hello, ReAct Langchain!!")
    tools = [get_text_length]

    template = """
        Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: {agent_scratchpad}
        """

    prompt_template = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([i.name for i in tools])
    )

    llm = chat_models.AzureChatOpenAI(
        model="gpt-4-32k",
        temperature=0,
        azure_endpoint="https://oceanfreightailabs.openai.azure.com/",
        azure_deployment="ocean_gpt4_32k",
        callbacks= [ AgentsCallbackHandler()]
    )

    intermediate_steps = []

    agent = {"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])} | prompt_template | llm | output_parser_func()

    agent_output = ""
    while not isinstance(agent_output, AgentFinish):
        agent_output: Union[AgentAction, AgentFinish] = agent.invoke(
            {"input": "What is the length in character of the text DOG?", "agent_scratchpad": intermediate_steps})

        if isinstance(agent_output, AgentAction):
            tool_name = agent_output.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_output.tool_input
            observation = tool_to_use.func(str(tool_input))
            print(f"{observation=}")
            intermediate_steps.append((agent_output, str(observation)))

    if isinstance(agent_output, AgentFinish):
        print(agent_output.return_values.get("output"))







