"""
https://langchain-ai.github.io/langgraph/agents/overview/
https://langchain-ai.github.io/langgraph/agents/agents/
https://langchain-ai.github.io/langgraph/tutorials/workflows/#pre-built
https://python.langchain.com/docs/how_to/custom_tools/
"""

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from shared.lc_llm import get_lc_llm
from pprint import pprint


@tool
def basic_calculator_tool(
    operation: Literal["add", "subtract", "multiply", "divide"], x: float, y: float
) -> int:
    """
    A tool for basic arithmetic operations between two numbers.
    """
    if operation == "add":
        return x + y
    elif operation == "subtract":
        return x - y
    elif operation == "multiply":
        return x * y
    elif operation == "divide":
        return x / y


def main():
    llm = get_lc_llm()

    system_prompt = """You are a helpful assistant in place to help with arithmetic."""
    user_prompt = (
        "What tools do you have access to? I want to multiply 632 and 721."  # 455672
    )

    print(f"Prompt: {user_prompt}")
    print("-" * 100)

    print("LLM without tools:")
    print(llm.invoke(user_prompt).content)
    print("-" * 100)

    print("Agent with tools:")
    tools = [basic_calculator_tool]

    pre_built_agent = create_react_agent(llm, tools=tools, prompt=system_prompt)

    messages = [HumanMessage(content=user_prompt)]
    messages = pre_built_agent.invoke({"messages": messages})
    for m in messages["messages"]:
        pprint(m, indent=4)

    """
    The LLM has the ability to stop it's output with a 'finish_reason' of 'tool_calls'. 
    This is different than the 'stop' keyword, which is used to stop the LLM's output.
    """


if __name__ == "__main__":
    main()
