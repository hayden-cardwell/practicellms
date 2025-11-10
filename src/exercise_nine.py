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


class StepCategory(BaseModel):
    """
    Pydantic Model for classifying the current step category.
    """

    category: Literal["research", "draft", "review", "formatting", "other"]


class StepSummary(BaseModel):
    """
    A compact rolling summary for a completed step.
    """

    step: str
    summary: str


class GraphState(BaseModel):
    """
    Pydantic Model defining the state for the graph.
    """

    # Messages are added to the state by the add_messages function.
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # Planner is a list of steps to be taken
    plan: Optional[list[str]] = None

    # Current step being executed
    current_step: Optional[str] = None

    # Current step category (classified once in prepare_step)
    current_step_category: Optional[
        Literal["research", "draft", "review", "formatting", "other"]
    ] = None

    # Rolling summaries from executing steps (most recent N only)
    artifacts: Annotated[list[StepSummary], add] = []

    # Report is the working report being written
    report: Optional[str] = None


# NODES
def gather_topic(state: GraphState) -> dict:
    """Node for gathering the report topic from the user."""

    user_topic = interrupt("What would you like me to write a report about?")

    return {"messages": [HumanMessage(content=user_topic)]}


def plan_report(state: GraphState, LLM) -> dict:
    """Node for planning the report."""

    system_message = SystemMessage(
        content="""Create a concise plan (5–8 steps) to write the report.
        Each step should be specific and actionable.
        Include research, drafting, review, and finalization as appropriate.
        Output one step per line, no bullets or numbers."""
    )

    llm_query_to_user = LLM.invoke([system_message, *state.messages])

    # Simply split by newlines and filter out empty lines
    plan_steps = [
        line.strip() for line in llm_query_to_user.content.split("\n") if line.strip()
    ]
    print(f"📝 Plan created with {len(plan_steps)} step(s)")
    return {"plan": plan_steps}


def prepare_step(state: GraphState, LLM) -> dict:
    """Node for preparing the next step by popping it from the plan and classifying it."""

    assert state.plan, "Plan is required to prepare a step."
    assert len(state.plan) > 0, "Plan must have at least one step."

    step = state.plan.pop(0)
    print(f"Preparing step: {step}")

    # Classify the step once to reduce downstream LLM calls
    classify_msg = SystemMessage(
        content=f"""Classify the plan step into one category:
        research | draft | review | formatting | other

        Step: {step}
        Answer with only the category."""
    )
    step_category = LLM.with_structured_output(StepCategory).invoke(
        [
            classify_msg,
            *state.messages,
        ]
    )
    print(f"🔎 Step category: {step_category.category}")

    return {
        "plan": state.plan,
        "current_step": step,
        "current_step_category": step_category.category,
    }


def execute_plan_step(state: GraphState, LLM) -> dict:
    """Node for executing the current plan step using available context and search results."""

    assert state.current_step, "Current step is required to execute a plan step."

    print(f"🚀 Executing step: {state.current_step}")
    print(f"   Category: {state.current_step_category}")

    # Prepare brief context from previous summaries
    context_summary = ""
    if state.artifacts:
        recent = state.artifacts[-3:]
        context_summary = "\n\nRecent step summaries:\n" + "\n".join(
            [f"- {s.step}: {s.summary[:200]}" for s in recent]
        )

    # Build the system message for executing this step
    current_report = (
        f"\n\nCurrent report draft:\n{state.report}\n" if state.report else ""
    )

    # If research step, generate 1-2 queries and fetch results
    search_items = []
    if state.current_step_category == "research":
        query_msg = SystemMessage(
            content=f"""Generate 1-2 concise web queries for this step (one per line). Step: {state.current_step}"""
        )
        queries = LLM.with_structured_output(SearchToolRequest).invoke(
            [query_msg, *state.messages]
        )
        print(f"🌐 Queries: {queries.queries}")
        search = DuckDuckGoSearchResults(max_results=5)
        for q in queries.queries:
            result = search.invoke(q)
            if isinstance(result, list):
                search_items.extend(result)
            else:
                search_items.append(result)
        print(f"🔎 Collected {len(search_items)} search result items")

    system_message = SystemMessage(
        content=f"""Execute the plan step.

        Step: {state.current_step}
        Category: {state.current_step_category}
        {context_summary}
        {current_report}
        Search results available: {len(search_items)} items

        Output only the result for this step.
        - If research: summarize findings in 2-3 sentences.
        - If draft: write the section content.
        - If review: provide concise improvement notes.
        """
    )

    # Get LLM to process the step
    step_output = LLM.invoke([system_message, *state.messages])

    print(f"🧩 Step output preview: {step_output.content[:200]}...")

    # Create a compact summary artifact
    artifacts = [
        StepSummary(step=state.current_step, summary=step_output.content[:300])
    ]
    print(f"✅ Step '{state.current_step}' completed.")

    # Update the report only for drafting steps
    updated_report = state.report or ""
    if state.current_step_category == "draft":
        updated_report += f"\n\n--- {state.current_step} ---\n{step_output.content}"
        print("📝 Appended drafted content to report")

    # Add the step output as a message to the conversation
    return {
        "artifacts": artifacts,
        "current_step": None,  # Clear current step after execution
        "current_step_category": None,
        "report": updated_report if updated_report != state.report else state.report,
        "messages": [
            AIMessage(content=f"Completed: {state.current_step}\n{step_output.content}")
        ],
    }


# (search decision is now implicit via current_step_category set in prepare_step)


# MAIN LOOP
def run_loop(lg_graph) -> None:
    # Define the thread ID for the conversation, could be random
    thread = {"configurable": {"thread_id": "1"}, "recursion_limit": 100}

    # Context always starts with pre-defined AI message
    command = {
        "messages": AIMessage(
            content="""Hello! I'm your AI report writing assistant. What can I write a report for you about today?"""
        )
    }

    while True:
        # Events are updates to the state
        for event in lg_graph.stream(
            command,
            thread,
            stream_mode="updates",
        ):
            # pprint(event)
            # print("-" * 80)

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
    g.add_node("prepare_step", partial(prepare_step, LLM=LLM))
    g.add_node("execute_plan_step", partial(execute_plan_step, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "gather_topic")
    g.add_edge("gather_topic", "plan_report")
    g.add_edge("plan_report", "prepare_step")

    # After preparing a step, execute it directly (search is handled inside execute when needed)
    g.add_edge("prepare_step", "execute_plan_step")

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    run_loop(lg_graph)


if __name__ == "__main__":
    main()
