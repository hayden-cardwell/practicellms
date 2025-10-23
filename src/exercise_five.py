from datetime import datetime
from functools import partial
from pprint import pprint
from typing import Annotated, Literal, Optional, Sequence

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
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


class SearchSummary(BaseModel):
    """
    Pydantic Model for the summary of the search results.
    """

    summary: str


class UserSatisfaction(BaseModel):
    """
    Pydantic Model for the user's satisfaction with the result.
    """

    user_satisfaction: Literal["satisfied", "unsatisfied"]


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

    # The summary of the search results.
    search_summary: Optional[str] = None

    # The calculator tool request is the request to the calculator tool.
    calculator_tool_request: Optional[CalculatorToolRequest] = None

    # Result is the final answer to the user's prompt.
    result: Optional[float] = None

    # The user's satisfaction with the result.
    user_satisfaction: Optional[UserSatisfaction] = None


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
        content=f"""You are a web-search expert assistant. For each user message, produce up to three concise, high-quality search queries most likely to return relevant and authoritative results.

    Important — use conversation context & user feedback:
    • Use the current conversation (the user's latest message, prior user messages, and assistant replies in this session) to infer intent, constraints, and any previously attempted queries.
    • If the user has given feedback about earlier queries (e.g., "narrow it", "more recent", "use academic sources", "exclude site:example.com", "try plain-language"), apply that feedback when generating new queries.
    • If the user's last message asks for multiple interpretations or asks for comparison/alternatives, ensure the queries cover those interpretations.

    Rules:
    1. Output 1–3 queries only. Place each query on its own line and include no additional commentary or explanation.
    2. Do NOT simply repeat or lightly rephrase the user's prompt. Translate the user's intent into focused search terms that capture key entities, constraints, and desired result types (e.g., tutorial, statistics, site:, filetype:).
    3. Use the current date ({datetime.now().strftime('%Y-%m-%d')}) to prefer recency when appropriate — explicitly include years or date ranges in queries for time-sensitive requests.
    4. Keep queries short and effective (roughly 3–12 words). Prefer keywords and search operators (quotes, site:, filetype:, OR, -, *) when they increase precision.
    5. When the user's request is ambiguous, provide queries that cover the most likely interpretations (e.g., definition vs how-to vs comparison), but still limit to at most three queries.
    6. Avoid stopwords, unnecessary punctuation, and full sentences unless required for exact-match (quoted) searches.
    """
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


def summarize_search_results(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant responsible for summarizing the results of search tool calls.\n\n"
            "Instructions:\n"
            "1. Use the provided search tool results to create a clear summary.\n"
            "2. Remove any duplicate or redundant information.\n"
            "3. Prioritize signal over noise, but do not lose any important information.\n"
            "4. You can be very verbose. The user will not be reading the summary, so you can be as detailed as you want.\n"
            "5. If a previous search summary exists, update or refine it using the new results.\n"
            "6. If no previous summary exists, generate a new summary based solely on the current results."
        )
    )

    llm_structured_output = LLM.with_structured_output(SearchSummary)
    search_summary_result = llm_structured_output.invoke(
        [system_message, *state.messages]
    )

    # Remove the search tool results from the messages
    messages = [m for m in state.messages if "Search tool results" not in m.content]

    # Remove the previous search summary from the messages
    messages = [m for m in messages if "Search summary" not in m.content]

    # Add the search summary to the messages
    messages = [
        *messages,
        AIMessage(content=f"Search summary: {search_summary_result.summary}"),
    ]

    return {"messages": messages, "search_summary": search_summary_result.summary}


def finalize_result(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to synthesize the results of previous tool calls and the search summary.
        Use the results of any previous tool calls to provide a final answer to the user's prompt.
        If the user's prompt involved a math calculation, use the result of the calculator tool.
        If the user's prompt involved searching the web, use the results of the search tool and the search summary.
        """
    )

    messages = [system_message, *state.messages]
    final_message = LLM.invoke(messages)
    return {"messages": [final_message]}


def check_user_satisfaction(state: GraphState, LLM) -> dict:
    """
    A node to check if the user is satisfied with the result.
    Waits for user input to confirm satisfaction or dissatisfaction.
    """

    feedback = interrupt("Are you satisfied with the result?")

    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to check if the user is satisfied with the result.
        """
    )

    llm_structured_output = LLM.with_structured_output(UserSatisfaction)
    user_satisfaction_result = llm_structured_output.invoke(
        [system_message, *state.messages, feedback]
    )

    messages = [*state.messages, HumanMessage(content=feedback)]

    print(f"User satisfaction result: {user_satisfaction_result}")
    return {"messages": messages, "user_satisfaction": user_satisfaction_result}


def user_satisfaction_edge(state: GraphState):
    return state.user_satisfaction.user_satisfaction


def run_loop(lg_graph: StateGraph) -> None:
    # Define the thread ID for the conversation, could be random
    thread = {"configurable": {"thread_id": "1"}}

    # Initial message from the user
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting loop.")
        return

    # default query if no input is provided
    if user_input == "":
        print("Using default query of 'Stock market results yesterday?'")
        user_input = "Stock market results yesterday?"

    # Create the initial command with the user's input
    command = {"messages": HumanMessage(content=user_input)}

    while True:
        # Events are updates to the state
        for event in lg_graph.stream(command, thread, stream_mode="updates"):
            pprint(event)
            print("-" * 80)

            # If the graph yields an end event, we're done
            if event.get("end"):
                print("✅ Conversation complete.")
                return

            # If the graph yields an interrupt event, we need to ask the user for feedback
            if "__interrupt__" in event:
                # Below prints the interrupt message that we define in the check_user_satisfaction node
                interrupt_msg = event["__interrupt__"][0].value
                print(interrupt_msg)
                user_feedback = input("Your response: ")

                # Using the feedback provided, we can resume the graph with the new command
                command = Command(resume=user_feedback)
                break

        else:
            raise ValueError("No interrupt event received")


def main():
    LLM = get_lc_llm()
    checkpoint_saver = InMemorySaver()

    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("determine_question_type", partial(determine_question_type, LLM=LLM))
    g.add_node("call_calculator_tool", partial(call_calculator_tool, LLM=LLM))
    g.add_node("finalize_result", partial(finalize_result, LLM=LLM))
    g.add_node("call_search_tool", partial(call_search_tool, LLM=LLM))
    g.add_node("summarize_search_results", partial(summarize_search_results, LLM=LLM))
    g.add_node("check_user_satisfaction", partial(check_user_satisfaction, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "determine_question_type")
    g.add_conditional_edges(
        "determine_question_type",
        question_type_edge,
        {"math": "call_calculator_tool", "other": "call_search_tool"},
    )
    g.add_edge("call_calculator_tool", "finalize_result")
    g.add_edge("call_search_tool", "summarize_search_results")
    g.add_edge("summarize_search_results", "finalize_result")
    g.add_edge("finalize_result", "check_user_satisfaction")
    g.add_conditional_edges(
        "check_user_satisfaction",
        user_satisfaction_edge,
        {"satisfied": END, "unsatisfied": "determine_question_type"},
    )

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    run_loop(lg_graph)


if __name__ == "__main__":
    main()
