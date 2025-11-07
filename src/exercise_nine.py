from datetime import datetime
from functools import partial
from operator import add
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

    # Planner is a list of steps to be taken
    plan: Optional[list[str]] = None

    # The queries to the search tool.
    search_queries: Annotated[list[str], add] = []

    # The results from the search tool.
    search_results: Annotated[list, add] = []

    # Report is the working report being written
    report: Optional[str] = None


# TOOLS
def search_tool(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are a web-search expert. 
        For each user message, output 1–3 concise search queries (one per line) and nothing else.
        Use conversation context and any prior query feedback. 
        Prefer recent results — include years/ranges when appropriate (Current date: {datetime.now().strftime('%Y-%m-%d')}).
        Make queries 3–12 words, use operators (quotes, site:, filetype:, OR, -, *) for precision. 
        If ambiguous, cover up to three likely interpretations. 
        Do not repeat the user's prompt.
        Keep in mind that you may have already called the search tool, and here's the previous search queries (if they exist): {state.search_queries if state.search_queries else "None"}
        Answer with a list of search queries.
        """
    )

    llm_structured_output = LLM.with_structured_output(SearchToolRequest)
    search_queries = llm_structured_output.invoke([system_message, *state.messages])

    search = DuckDuckGoSearchResults(max_results=5)
    search_results = [search.invoke(q) for q in search_queries.queries]

    return {
        "search_queries": search_queries.queries,
        "search_results": search_results,
    }


# NODES
def gather_topic(state: GraphState) -> dict:
    """Node for gathering the report topic from the user."""

    user_topic = interrupt("What would you like me to write a report about?")

    return {"messages": [HumanMessage(content=user_topic)]}


def plan_report(state: GraphState, LLM) -> dict:
    """Node for planning the report."""

    system_message = SystemMessage(
        content="""You are a helpful assistant that creates detailed plans for writing reports.
        
        Based on the user's request, create a comprehensive list of steps needed to write the report.
        Each step should be specific and actionable.
        
        Create between 5 and 15 steps for the plan.
        
        Your plan should include steps such as:
        - Research specific topics or gather information
        - Draft specific sections (introduction, body, conclusion, etc.)
        - Review and revise content
        - Finalize formatting and structure
        
        Output your plan with each step on a new line, without any bullet points or numbering.
        For example:
        Research the history of [topic]
        Draft the introduction section
        Draft the methodology section
        Review and revise for clarity
        Finalize formatting and citations
        
        Be specific about what needs to be researched, drafted, or revised in each step.
        """
    )

    llm_query_to_user = LLM.invoke([system_message, *state.messages])

    # Simply split by newlines and filter out empty lines
    plan_steps = [
        line.strip() for line in llm_query_to_user.content.split("\n") if line.strip()
    ]

    return {"plan": plan_steps}


def execute_plan_step(state: GraphState, LLM) -> dict:
    """Node for executing a plan step."""

    system_message = SystemMessage(
        content="""You are a helpful assistant that executes a plan step.
        """
    )


def run_loop(lg_graph) -> None:
    # Define the thread ID for the conversation, could be random
    thread = {"configurable": {"thread_id": "1"}}

    # Context always starts with pre-defined AI message
    command = {
        "messages": AIMessage(
            content="""Hello! I'm your AI report writing assistant. What can I write a report for you about today?"""
        )
    }

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
                # Below prints the interrupt message that we define in the gather_information node
                interrupt_msg = event["__interrupt__"][0].value
                print(interrupt_msg)
                user_input = input("Your response: ")

                # Using the input provided, we can resume the graph with the new command
                command = Command(resume=user_input)
                break

        else:
            break


def main():
    LLM = get_lc_llm()
    checkpoint_saver = InMemorySaver()
    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("gather_topic", gather_topic)
    g.add_node("plan_report", partial(plan_report, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "gather_topic")
    g.add_edge("gather_topic", "plan_report")
    g.add_edge("plan_report", END)

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    run_loop(lg_graph)


if __name__ == "__main__":
    main()
