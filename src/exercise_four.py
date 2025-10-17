from functools import partial
from pprint import pprint
from typing import Literal, Optional
from typing import Annotated, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from shared.lc_llm import get_lc_llm


class QuestionType(BaseModel):
    """
    Pydantic Model defining the type of question the user is asking.
    """

    question_type: Literal["math", "other"]


class CalculatorToolRequest(BaseModel):
    """
    Pydantic Model for the calculator tool.
    """

    operation: Literal["add", "subtract", "multiply", "divide"]
    x: float
    y: float


class GraphState(BaseModel):
    """
    Pydantic Model defining the state for the graph.
    """

    # Messages are added to the state by the add_messages function.
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # The calculator tool request is the request to the calculator tool.
    calculator_tool_request: Optional[CalculatorToolRequest] = None

    # Determines the type of question the user is asking.
    question_type: Optional[QuestionType] = None

    # Result is the final answer to the user's prompt.
    result: Optional[float] = None


def calculator_tool(
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


def call_calculator_tool(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content="""You are a helpful assistant in place to help with arithmetic. You call the calculator tool to perform the math as necessary"""
    )

    llm_structured_output = LLM.with_structured_output(CalculatorToolRequest)
    calc_request_result = llm_structured_output.invoke(
        [system_message, *state.messages]
    )

    # Call the calculator tool
    calculator_tool_result = calculator_tool(
        operation=calc_request_result.operation,
        x=calc_request_result.x,
        y=calc_request_result.y,
    )

    messages = [
        *state.messages,
        AIMessage(content=f"Calculator tool result: {calculator_tool_result}"),
    ]

    return {
        "messages": messages,
        "calculator_tool_request": calc_request_result,
        "result": calculator_tool_result,
    }


def finalize_result(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to help with arithmetic. Trust the result of the calculator tool if one has been provided. If there is any other information that needs to be provided, provide it, but trust the result of the calculator tool if one has been provided."""
    )

    messages = [system_message, *state.messages]
    final_message = LLM.invoke(messages)
    return {"messages": [final_message]}


def main():
    USER_PROMPT = (
        "What tools do you have access to? I want to multiply 632 and 721."  # 455672
    )
    # USER_PROMPT = "What is the capital of France?"

    LLM = get_lc_llm()

    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("determine_question_type", partial(determine_question_type, LLM=LLM))
    g.add_node("call_calculator_tool", partial(call_calculator_tool, LLM=LLM))
    g.add_node("finalize_result", partial(finalize_result, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "determine_question_type")
    g.add_conditional_edges(
        "determine_question_type",
        question_type_edge,
        {"math": "call_calculator_tool", "other": END},
    )
    g.add_edge("call_calculator_tool", "finalize_result")
    g.add_edge("finalize_result", END)

    lg_graph = g.compile()

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    # Invoke
    state = lg_graph.invoke({"messages": HumanMessage(content=USER_PROMPT)})
    pprint(state, indent=4)


if __name__ == "__main__":
    main()
