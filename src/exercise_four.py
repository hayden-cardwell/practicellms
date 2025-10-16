"""
https://python.langchain.com/docs/how_to/custom_tools/
"""

from functools import partial
from pprint import pprint
from typing import Literal, Optional
from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from shared.lc_llm import get_lc_llm


class QuestionType(BaseModel):
    """
    Pydantic Model defining the type of question the user is asking.
    """

    question_type: Literal["math", "other"]


class GraphState(BaseModel):
    """
    Pydantic Model defining the state for the graph.
    """

    # Messages are added to the state by the add_messages function.
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Determines the type of question the user is asking.
    question_type: Optional[QuestionType] = None

    # Result is the final answer to the user's prompt.
    result: Optional[float] = None


@tool
def basic_calculator_tool(
    operation: Literal["add", "subtract", "multiply", "divide"], x: float, y: float
) -> float:
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


# Nodes
def determine_question_type(state: GraphState, LLM) -> dict:
    prompt = f"""You are a question router.
    You are given a user prompt and you need to determine if the prompt requires a math calculation.
    If it does, return "math". If it doesn't, return "other". Answer with "math" or "other" ONLY.
    Your User Prompt: {state.messages[-1].content}"""

    llm_structured_output = LLM.with_structured_output(QuestionType)
    result = llm_structured_output.invoke(prompt)
    return {"question_type": result}


def question_type_edge(state: GraphState):
    return state.question_type.question_type


def do_math(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content="""You are a helpful assistant in place to help with arithmetic. You call the calculator tool to perform the math as necessary"""
    )

    response = LLM.invoke([system_message, *state.messages])
    return {"messages": [response]}


def needs_tool_call(state: GraphState) -> str:
    last = state.messages[-1]
    return (
        "tool_node"
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None)
        else END
    )


def main():
    USER_PROMPT = (
        "What tools do you have access to? I want to multiply 632 and 721."  # 455672
    )
    # USER_PROMPT = "What is the capital of France?"

    TOOLS = [basic_calculator_tool]
    LLM = get_lc_llm()
    LLM_WITH_TOOLS = LLM.bind_tools(TOOLS)  # Runnable
    tool_node = ToolNode(TOOLS)  # Special node required to run tools.

    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("determine_question_type", partial(determine_question_type, LLM=LLM))
    g.add_node("do_math", partial(do_math, LLM=LLM_WITH_TOOLS))
    g.add_node("tool_node", tool_node)

    # Add edges to the graphs
    g.add_edge(START, "determine_question_type")
    g.add_conditional_edges(
        "determine_question_type",
        question_type_edge,
        {
            "math": "do_math",
            "other": END,
        },
    )
    g.add_conditional_edges(
        "do_math",
        needs_tool_call,
        {
            "tool_node": "tool_node",
            END: END,
        },
    )
    g.add_edge("tool_node", "do_math")

    lg_graph = g.compile()

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    # Invoke
    state = lg_graph.invoke({"messages": HumanMessage(content=USER_PROMPT)})
    pprint(state, indent=4)


if __name__ == "__main__":
    main()
