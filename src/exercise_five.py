from datetime import datetime
from functools import partial
from pprint import pprint
from typing import Literal, Optional
from typing import Annotated, Sequence

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
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


class SearchToolRequest(BaseModel):
    """
    Pydantic Model for the search tool.
    """

    queries: list[str]


class GraphState(BaseModel):
    """
    Pydantic Model defining the state for the graph.
    """

    # Messages are added to the state by the add_messages function.
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Determines the type of question the user is asking.
    question_type: Optional[QuestionType] = None

    # The query to the search tool.
    search_query: Optional[SearchToolRequest] = None

    # The results of the search tool.
    search_results: Optional[list] = None

    # The calculator tool request is the request to the calculator tool.
    calculator_tool_request: Optional[CalculatorToolRequest] = None

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


def search_tool(queries: str) -> list:
    search = DuckDuckGoSearchResults(max_results=5)
    results = [search.invoke(q) for q in queries]
    return results


# Nodes
def determine_question_type(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are a question router.
    You are given a user prompt and you need to determine if the prompt requires a math calculation.
    If it does, return "math". If it doesn't, return "other". Answer with "math" or "other" ONLY. 
    The user prompts involving math will be sent to a calculator tool, and the user prompts involving other information will be sent to a search tool.
    """
    )

    llm_structured_output = LLM.with_structured_output(QuestionType)
    question_type_result = llm_structured_output.invoke(
        [system_message, *state.messages]
    )
    return {"question_type": question_type_result}


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


def call_search_tool(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=(
            f"""You are a web search expert assistant. 
            Given a user's message, generate a clear and effective search engine query that will best help answer the user's question. 
            Do not simply repeat or rephrase the user's prompt. Carefully consider the user's intent and provide a concise, high-quality query likely to yield relevant and informative results.
            Provide five search queries.
            The current date is {datetime.now().strftime('%Y-%m-%d')}"""
        )
    )

    llm_structured_output = LLM.with_structured_output(SearchToolRequest)
    search_request_result = llm_structured_output.invoke(
        [system_message, *state.messages]
    )

    search_results = search_tool(search_request_result.queries)

    messages = [
        *state.messages,
        AIMessage(content=f"Search tool results: {search_results}"),
    ]

    return {
        "messages": messages,
        "search_query": search_request_result,
        "search_results": search_results,
    }


def finalize_result(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to synthesize the results of previous tool calls.
        Use the results of any previous tool calls to provide a final answer to the user's prompt.
        If the user's prompt involved a math calculation, use the result of the calculator tool.
        If the user's prompt involved searching the web, use the results of the search tool.
        """
    )

    messages = [system_message, *state.messages]
    final_message = LLM.invoke(messages)
    return {"messages": [final_message]}


def main():
    LLM = get_lc_llm()

    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("determine_question_type", partial(determine_question_type, LLM=LLM))
    g.add_node("call_calculator_tool", partial(call_calculator_tool, LLM=LLM))
    g.add_node("finalize_result", partial(finalize_result, LLM=LLM))
    g.add_node("call_search_tool", partial(call_search_tool, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "determine_question_type")
    g.add_conditional_edges(
        "determine_question_type",
        question_type_edge,
        {"math": "call_calculator_tool", "other": "call_search_tool"},
    )
    g.add_edge("call_calculator_tool", "finalize_result")
    g.add_edge("call_search_tool", "finalize_result")
    g.add_edge("finalize_result", END)

    lg_graph = g.compile()

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    # TEST MULTIPLE PROMPTS
    user_prompts = [
        "What tools do you have access to? I want to multiply 632 and 721.",  # 455672
        "What is the capital of France?",
        "What happened yesterday in the stock market?",
    ]

    for prompt in user_prompts:
        state = lg_graph.invoke({"messages": HumanMessage(content=prompt)})

        pprint(state, indent=4)

        print("-" * 100)


if __name__ == "__main__":
    main()
