import argparse
from functools import partial
from operator import add
from typing import Annotated, Literal, Optional, Sequence
from pprint import pprint
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from shared.lc_llm import get_lc_llm


class PlanStep(BaseModel):
    """A single step in the plan."""

    id: int
    desc: str
    done: bool


class Plan(BaseModel):
    """A plan for the goal."""

    steps: list[PlanStep]


class GraphState(BaseModel):
    """
    Pydantic Model defining the state for the graph.
    """

    # Messages are added to the state by the add_messages function.
    messages: Annotated[Sequence[BaseMessage], add_messages]

    goal: Optional[str] = None
    plan: Optional[Plan] = None
    cursor: Optional[int] = None
    artifacts: Annotated[list[str], add] = []
    answer: Optional[str] = None
    error: Optional[str] = None


# NODES
def gather_goal(state: GraphState) -> dict:
    """Node for gathering the goal from the user via interrupt."""
    user_goal = interrupt("What would you like me to help you with today?")
    return {"goal": user_goal, "messages": [HumanMessage(content=user_goal)]}


def plan(state: GraphState, LLM) -> dict:
    """Node for planning the goal."""
    system_message = SystemMessage(
        content="""Create a concise plan (3–5 steps) for the goal.
        Each step should be specific and actionable.
        Include research, drafting, review, and finalization as appropriate.
        
        For each step provide:
        - id: sequential number starting from 1
        - desc: clear description of what to do
        - done: always false for new plans"""
    )

    llm_structured = LLM.with_structured_output(Plan)
    llm_output = llm_structured.invoke([system_message, *state.messages])
    return {"plan": llm_output, "cursor": 0}


def route_next(state: GraphState) -> str:
    """Route based on cursor position."""
    if state.cursor >= len(state.plan.steps):
        return "answer"
    return "act"


def act(state: GraphState, LLM) -> dict:
    """Node for acting on the current step."""
    system_message = SystemMessage(
        content=f"""You are a helpful assistant that acts on the current step: 
        {state.plan.steps[state.cursor].desc}"""
    )
    llm_output = LLM.invoke([system_message, *state.messages])
    return {
        "artifacts": [llm_output.content],
        "cursor": state.cursor + 1,
        "plan": {
            "steps": [
                {
                    "id": state.plan.steps[state.cursor].id,
                    "desc": state.plan.steps[state.cursor].desc,
                    "done": True,
                }
            ]
        },
    }


def answer(state: GraphState, LLM) -> dict:
    """Node for synthesizing artifacts into a final answer."""
    artifacts_context = (
        "\n".join(state.artifacts) if state.artifacts else "No artifacts collected."
    )

    system_message = SystemMessage(
        content=f"""Based on the following research and work collected during the planning process, 
        provide a comprehensive final answer to the user's goal: {state.goal}
        
        Context from work completed:
        {artifacts_context}"""
    )

    llm_output = LLM.invoke([system_message, *state.messages])
    return {"answer": llm_output.content}


# MAIN LOOP
def parse_cli_args() -> argparse.Namespace:
    """Parse CLI flags for configuring the agent."""

    parser = argparse.ArgumentParser(
        description="Exercise 9: multi-turn planning with retries and safety cap."
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum number of steps to execute before returning partial results.",
    )

    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be a positive integer")

    return args


def run_loop(lg_graph, max_steps: Optional[int]) -> None:
    # Define the thread ID for the conversation, could be random
    thread = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}

    # Context always starts with pre-defined AI message
    command = {
        "messages": AIMessage(
            content="""Hello! I'm your AI assistant. What can I help you with today?"""
        ),
    }

    if max_steps is not None:
        command["step_limit"] = max_steps
        print(f"🛑 Max steps set to {max_steps}")

    while True:
        # Events are updates to the state
        for event in lg_graph.stream(
            command,
            thread,
            stream_mode="updates",
        ):
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
                # user_input = input("Your response: ")
                user_input = "Compare the latest iphone to the latest pixel?"

                # Using the input provided, we can resume the graph with the new command
                command = Command(resume=user_input)
                break

        else:
            break


def main():
    args = parse_cli_args()
    LLM = get_lc_llm()
    checkpoint_saver = InMemorySaver()
    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("gather_goal", gather_goal)
    g.add_node("plan", partial(plan, LLM=LLM))
    g.add_node("act", partial(act, LLM=LLM))
    g.add_node("answer", partial(answer, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "gather_goal")
    g.add_edge("gather_goal", "plan")
    g.add_conditional_edges("plan", route_next)
    g.add_conditional_edges("act", route_next)
    g.add_edge("answer", END)

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    run_loop(lg_graph, args.max_steps)


if __name__ == "__main__":
    main()
