import argparse
from functools import partial
from operator import add
from typing import Annotated, Literal, Optional, Sequence

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ValidationError

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

    # Tool currently selected for the active step
    current_tool: Optional[str] = None

    # Whether the active step should be retried with a different tool
    retry_requested: bool = False

    # Maximum number of steps allowed in this run (None means unlimited)
    step_limit: Optional[int] = None

    # Count of completed steps in this run
    steps_completed: int = 0


TOOL_SEQUENCES = {
    "research": ["search_and_write", "llm_only"],
    "default": ["llm_only"],
}


TOOL_LABELS = {
    "search_and_write": "web search + synthesis",
    "llm_only": "LLM-only drafting",
}


def get_tool_sequence(category: Optional[str]) -> list[str]:
    """Return the ordered tooling sequence for the given category."""

    return TOOL_SEQUENCES["research"] if category == "research" else TOOL_SEQUENCES["default"]


def initial_tool(category: Optional[str]) -> str:
    """Choose the primary tool for a step."""

    return get_tool_sequence(category)[0]


def next_tool(category: Optional[str], failed_tool: Optional[str]) -> Optional[str]:
    """Return the next tool to try after the provided tool fails."""

    if not failed_tool:
        return None

    sequence = get_tool_sequence(category)
    if failed_tool not in sequence:
        return None

    idx = sequence.index(failed_tool)
    if idx + 1 < len(sequence):
        return sequence[idx + 1]
    return None


def describe_tool(tool: Optional[str]) -> str:
    """Return a friendly label for the tool identifier."""

    if not tool:
        return "unspecified tool"
    return TOOL_LABELS.get(tool, tool)


def request_retry_with_fallback(state: GraphState, failed_tool: str, error: Exception) -> dict:
    """Prepare state updates to retry the current step with a fallback tool."""

    fallback = next_tool(state.current_step_category, failed_tool)
    if not fallback:
        raise RuntimeError(
            f"Step '{state.current_step}' failed with {failed_tool} and no fallback is available."
        ) from error

    print(
        f"⚠️ Tool '{describe_tool(failed_tool)}' failed for step '{state.current_step}'."
    )
    print(f"   Retrying with {describe_tool(fallback)}")

    return {
        "current_tool": fallback,
        "retry_requested": True,
        "messages": [
            AIMessage(
                content=(
                    f"Hit an issue with {describe_tool(failed_tool)} while working on"
                    f" '{state.current_step}'. Switching to {describe_tool(fallback)}"
                    " to retry this step."
                )
            )
        ],
    }


def execution_progress_edge(state: GraphState) -> Literal["retry", "limit", "continue", "done"]:
    """Route execution based on retries, safety caps, and remaining plan items."""

    if state.retry_requested:
        return "retry"

    if (
        state.step_limit is not None
        and state.steps_completed >= state.step_limit
        and state.plan
        and len(state.plan) > 0
    ):
        return "limit"

    if state.plan:
        return "continue"

    return "done"


# NODES
def gather_topic(state: GraphState) -> dict:
    """Node for gathering the report topic from the user."""

    user_topic = interrupt("What would you like me to write a report about?")

    return {"messages": [HumanMessage(content=user_topic)]}


def plan_report(state: GraphState, LLM) -> dict:
    """Node for planning the report."""

    def normalize_plan_line(line: str) -> str:
        """Strip bullet characters / numbering from a plan line."""

        stripped = line.strip()
        stripped = stripped.lstrip("-•* \t").strip()

        numeric_prefix = ""
        while stripped and stripped[0].isdigit():
            numeric_prefix += stripped[0]
            stripped = stripped[1:]

        if numeric_prefix and stripped and stripped[0] in {".", ")", ":"}:
            stripped = stripped[1:]

        return stripped.strip()

    system_message = SystemMessage(
        content="""Create a concise plan (5–8 steps) to write the report.
        Each step should be specific and actionable.
        Include research, drafting, review, and finalization as appropriate.
        Return the steps as a bullet list (one per line, prefix with '- ')."""
    )

    llm_query_to_user = LLM.invoke([system_message, *state.messages])

    # Split by newlines, normalize bullet formatting, and filter out empty lines
    plan_steps = []
    for line in llm_query_to_user.content.split("\n"):
        cleaned = normalize_plan_line(line)
        if cleaned:
            plan_steps.append(cleaned)

    if not plan_steps:
        raise ValueError("LLM did not return any plan steps.")

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
    try:
        step_category = LLM.with_structured_output(StepCategory).invoke(
            [
                classify_msg,
                *state.messages,
            ]
        )
        category = step_category.category
    except (ValidationError, ValueError) as exc:
        print("⚠️ Structured classification failed, falling back to plain text.")
        fallback_msg = SystemMessage(
            content=(
                "Respond with only one word category (research, draft, review, formatting, other)."
                f" Step: {step}"
            )
        )
        fallback_response = LLM.invoke([fallback_msg, *state.messages])
        candidate = fallback_response.content.strip().lower().split()
        category = candidate[0] if candidate else "other"
        if category not in {"research", "draft", "review", "formatting", "other"}:
            category = "other"
        print(f"   Fallback category guess: {category}")
    else:
        print(f"🔎 Step category: {category}")

    tool_choice = initial_tool(category)
    print(f"🧰 Selected tool: {describe_tool(tool_choice)}")

    return {
        "plan": state.plan,
        "current_step": step,
        "current_step_category": category,
        "current_tool": tool_choice,
        "retry_requested": False,
    }


def execute_plan_step(state: GraphState, LLM) -> dict:
    """Node for executing the current plan step using available context and search results."""

    assert state.current_step, "Current step is required to execute a plan step."

    print(f"🚀 Executing step: {state.current_step}")
    print(f"   Category: {state.current_step_category}")
    tool_choice = state.current_tool or initial_tool(state.current_step_category)
    print(f"   Tool mode: {describe_tool(tool_choice)}")

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

    # If the selected tool requires search, fetch supporting context
    search_items = []
    if tool_choice == "search_and_write":
        try:
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
        except Exception as exc:  # noqa: BLE001
            return request_retry_with_fallback(state, tool_choice, exc)

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
    try:
        step_output = LLM.invoke([system_message, *state.messages])
    except Exception as exc:  # noqa: BLE001
        return request_retry_with_fallback(state, tool_choice, exc)

    print(f"🧩 Step output preview: {step_output.content[:200]}...")

    # Create a compact summary artifact
    artifacts = [
        StepSummary(step=state.current_step, summary=step_output.content[:300])
    ]
    print(f"✅ Step '{state.current_step}' completed.")

    # Update the report only for drafting steps
    updated_report = state.report or ""
    report_changed = False
    if state.current_step_category == "draft":
        updated_report += f"\n\n--- {state.current_step} ---\n{step_output.content}"
        report_changed = True
        print("📝 Appended drafted content to report")

    # Add the step output as a message to the conversation
    return {
        "artifacts": artifacts,
        "current_step": None,  # Clear current step after execution
        "current_step_category": None,
        "report": updated_report if report_changed else state.report,
        "current_tool": None,
        "retry_requested": False,
        "steps_completed": state.steps_completed + 1,
        "messages": [
            AIMessage(content=f"Completed: {state.current_step}\n{step_output.content}")
        ],
    }


def finalize_report(state: GraphState) -> dict:
    """Summarize progress and provide the latest draft or partial results."""

    remaining_steps = state.plan or []
    limit_hit = (
        state.step_limit is not None
        and state.steps_completed >= state.step_limit
        and len(remaining_steps) > 0
    )

    if limit_hit:
        status = (
            f"Safety cap reached after {state.steps_completed} step(s); "
            "returning partial progress."
        )
    elif remaining_steps:
        status = "Concluding early with remaining plan items listed below."
    else:
        status = "All planned steps completed."

    recent_artifacts = state.artifacts[-5:] if state.artifacts else []
    artifact_lines = "\n".join(
        [f"- {artifact.step}: {artifact.summary[:200]}" for artifact in recent_artifacts]
    )
    artifact_block = (
        f"\nRecent progress:\n{artifact_lines}" if recent_artifacts else "\nNo step summaries recorded."
    )

    remaining_block = ""
    if remaining_steps:
        remaining_block = "\n\nRemaining plan items:\n" + "\n".join(
            [f"- {step}" for step in remaining_steps]
        )

    report_body = state.report.strip() if state.report else "No draft content captured yet."

    final_message = (
        f"{status}\n\nReport draft:\n{report_body}{artifact_block}{remaining_block}"
    )

    print("\n📄 Final report summary:\n")
    print(final_message)

    return {
        "messages": [AIMessage(content=final_message)],
    }


# (search decision is now implicit via current_step_category set in prepare_step)


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
            content="""Hello! I'm your AI report writing assistant. What can I write a report for you about today?"""
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
    args = parse_cli_args()
    LLM = get_lc_llm()
    checkpoint_saver = InMemorySaver()
    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("gather_topic", gather_topic)
    g.add_node("plan_report", partial(plan_report, LLM=LLM))
    g.add_node("prepare_step", partial(prepare_step, LLM=LLM))
    g.add_node("execute_plan_step", partial(execute_plan_step, LLM=LLM))
    g.add_node("finalize_report", finalize_report)

    # Add edges to the graphs
    g.add_edge(START, "gather_topic")
    g.add_edge("gather_topic", "plan_report")
    g.add_edge("plan_report", "prepare_step")

    # After preparing a step, execute it directly (search is handled inside execute when needed)
    g.add_edge("prepare_step", "execute_plan_step")

    g.add_conditional_edges(
        "execute_plan_step",
        execution_progress_edge,
        {
            "retry": "execute_plan_step",
            "continue": "prepare_step",
            "done": "finalize_report",
            "limit": "finalize_report",
        },
    )

    g.add_edge("finalize_report", END)

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    run_loop(lg_graph, args.max_steps)


if __name__ == "__main__":
    main()
