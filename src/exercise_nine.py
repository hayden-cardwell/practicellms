from functools import partial
from pprint import pprint
from typing import Annotated, Literal, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from shared.lc_llm import get_lc_llm


class GraphState(BaseModel):
    """
    Pydantic Model defining the state for the graph.
    """

    # Messages are added to the state by the add_messages function.
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Planner is a list of steps to be taken
    plan: Optional[list[str]] = None


# TOOLS


# NODES
def plan_report(state: GraphState, LLM) -> dict:
    """Node for planning the report."""

    system_message = SystemMessage(
        content="""You are a helpful assistant planning a report.
        You need to create a list of steps to take to write the report.
        Be specific and detailed.
        """
    )

    llm_query_to_user = LLM.invoke([system_message, *state.messages])

    return {"plan": llm_query_to_user.content}


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
    g.add_node("plan_report", partial(plan_report, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "plan_report")
    g.add_edge("plan_report", END)

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    run_loop(lg_graph)


if __name__ == "__main__":
    main()
