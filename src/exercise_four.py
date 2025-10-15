"""
https://python.langchain.com/docs/how_to/custom_tools/
"""

from pprint import pprint
from typing import Literal, Optional
from typing import Annotated, Sequence

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from functools import partial

from shared.lc_llm import get_lc_llm


# Tool Stuff
class MathInput(BaseModel):
    operation: Literal["add", "subtract", "multiply", "divide"]
    x: float
    y: float


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


class IsMathQuestionOutput(BaseModel):
    is_math_question: bool


# Workflow State
class GraphState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_math_question: Optional[IsMathQuestionOutput] = None

    result: Optional[float] = None


# Nodes
def is_math_question(state: GraphState, LLM) -> IsMathQuestionOutput:
    messages = state.messages
    llm_structured_output = LLM.with_structured_output(IsMathQuestionOutput)
    prompt = f"Is the following user prompt a math question? Answer with True or False ONLY. User Prompt: {state.messages[-1].content}"
    return llm_structured_output.invoke(prompt)


def math_exec(state: GraphState, LLM) -> GraphState:
    prompt = f"""You are a helpful assistant in place to help with arithmetic. 
    Your task is to extract the necessary information from the prompt and call the calculator tool.
    Your User Prompt: {state.messages[-1].content}"""
    return LLM.invoke(prompt)


# Conditional Edges
def math_question_edge(state: GraphState) -> bool:
    if state.is_math_question.is_math_question == True:
        return "math_exec", {"result": state.is_math_question.result}
    else:
        return END


def main():

    SYSTEM_PROMPT = """You are a helpful assistant in place to help with arithmetic."""
    USER_PROMPT = (
        "What tools do you have access to? I want to multiply 632 and 721."  # 455672
    )

    TOOLS = [basic_calculator_tool]
    LLM = get_lc_llm()
    LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)

    lg_graph_builder = StateGraph(GraphState)

    lg_graph_builder.add_node("is_math_question", partial(is_math_question, LLM=LLM))
    lg_graph_builder.add_node("math_exec", partial(math_exec, LLM=LLM_WITH_TOOLS))

    lg_graph_builder.add_edge(START, "is_math_question")
    lg_graph_builder.add_conditional_edges(
        "is_math_question",
        math_question_edge,
        {
            True: "math_exec",
            False: END,
        },
    )
    lg_graph_builder.add_edge("math_exec", END)

    lg_graph = lg_graph_builder.compile()

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    # Invoke
    state = lg_graph.invoke({"messages": HumanMessage(content=USER_PROMPT)})
    pprint(state, indent=4)


if __name__ == "__main__":
    main()
