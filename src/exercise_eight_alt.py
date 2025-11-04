from datetime import datetime
from functools import partial
from operator import add
import os
from pprint import pprint
from typing import Annotated, Literal, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from shared.lc_llm import get_lc_llm


class ReadyToComposeEmail(BaseModel):
    """
    Pydantic Model defining if the information gathered from the user is sufficient to compose an email.
    """

    ready_to_compose: bool


class Email(BaseModel):
    """
    Pydantic Model defining the email to send.
    """

    email_address: str
    subject: str
    body: str


class GraphState(BaseModel):
    """
    Pydantic Model defining the state for the graph.
    """

    # Messages are added to the state by the add_messages function.
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # If email is ready to be composed
    ready_to_compose_email: Optional[ReadyToComposeEmail] = None

    # The email to send
    email: Optional[Email] = None

    # Ready to send
    ready_to_send: Optional[bool] = None

    # Feedback on the email from the user
    email_feedback: Optional[str] = None


# TOOLS
def send_email(email: Email):
    """Mock tool mimicking the send_email function."""
    print("Sending the following email:")
    pprint(email)
    print("Email sent successfully.")
    return True


# NODES
def gather_information(state: GraphState, LLM) -> dict:
    """Node for gathering information from the user in furtherance of sending an email."""

    system_message = SystemMessage(
        content="""You are a helpful assistant gathering information to compose an email.
        You need to collect two pieces of information from the user:
        1. Who the email is for (the recipient's email address)
        2. What they want to say (the main message or purpose of the email)
        
        Be friendly and professional.
        Ask ONE question at a time to make the conversation natural and not overwhelming.
        Once you have both pieces of information, you will help draft the email with an appropriate subject line and well-formatted body."""
    )

    # Gen question
    llm_query_to_user = LLM.invoke([system_message, *state.messages])

    # Interrupt the graph to ask the user the question
    interrupt_msg = llm_query_to_user.content
    user_response = interrupt(interrupt_msg)

    return {
        "messages": [
            *state.messages,
            AIMessage(content=interrupt_msg),
            HumanMessage(content=user_response),
        ]
    }


def validate_information(state: GraphState, LLM) -> dict:
    """Node for validating the information gathered from the user to
    determine if we have enough context to send the email."""

    system_message = SystemMessage(
        content="""You are a helpful assistant validating the information gathered from the user.
        You need to validate the information gathered from the user and ensure it is correct and complete.
        Answer with a boolean value indicating if the information is sufficient to compose an email.
        """
    )

    llm_with_structured_output = LLM.with_structured_output(ReadyToComposeEmail)
    llm_output = llm_with_structured_output.invoke([system_message, *state.messages])

    return {
        "ready_to_compose_email": llm_output,
    }


def validate_information_edge(state: GraphState) -> Literal["complete", "incomplete"]:
    """Returns 'complete' if the information is valid, 'incomplete' otherwise."""
    if state.ready_to_compose_email.ready_to_compose:
        return "complete"
    else:
        return "incomplete"


def draft_email(state: GraphState, LLM) -> dict:
    """Node for drafting the email based on the information gathered from the user."""

    system_message = SystemMessage(
        content="""You are a helpful assistant drafting an email based on the information gathered from the user."""
    )

    llm_with_structured_output = LLM.with_structured_output(Email)
    llm_output = llm_with_structured_output.invoke([system_message, *state.messages])

    preformatted_message = AIMessage(
        content=f"I've drafted the following email:\n\n{llm_output}"
    )

    return {
        "email": llm_output,
        "messages": [
            *state.messages,
            preformatted_message,
        ],
    }


def show_email(state: GraphState) -> dict:
    """Node for showing the email to the user."""
    print("The following email will be sent:")
    pprint(state.email)

    while True:
        print("Would you like to send this email? (y/n)")
        user_input = input("Your response: ")

        if user_input.lower() == "y":
            return {"ready_to_send": True}
        elif user_input.lower() == "n":
            interrupt_msg = "What would you like to change about the email?"
            email_feedback = interrupt(interrupt_msg)
            return {
                "email_feedback": email_feedback,
                "ready_to_send": False,
                "messages": [
                    *state.messages,
                    AIMessage(content=interrupt_msg),
                    HumanMessage(content=email_feedback),
                    AIMessage(
                        content=f"Thank you for your feedback. I will update the email based on your feedback."
                    ),
                ],
            }
        else:
            print("Please enter 'y' or 'n'.")


def show_email_edge(state: GraphState) -> Literal["send", "update"]:
    """Returns 'send' if the user wants to send the email, 'update' if the user wants to update the email."""

    if state.ready_to_send:
        return "send"
    else:
        return "update"


def call_send_email(state: GraphState) -> dict:
    """Node for calling the send_email tool to send the email."""

    if send_email(state.email):
        return {
            "messages": [*state.messages, AIMessage(content="Email sent successfully.")]
        }
    else:
        return {
            "messages": [*state.messages, AIMessage(content="Failed to send email.")]
        }


def run_loop(lg_graph) -> None:
    # Define the thread ID for the conversation, could be random
    thread = {"configurable": {"thread_id": "1"}}

    # Context always starts with pre-defined AI message
    command = {
        "messages": AIMessage(
            content="""Hello! I'm your AI email assistant. 
            I'm here to help you compose and send an email. 
            I'll ask you a few questions to gather the information needed to draft your email.
            """
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
    g.add_node("gather_information", partial(gather_information, LLM=LLM))
    g.add_node("validate_information", partial(validate_information, LLM=LLM))
    g.add_node("draft_email", partial(draft_email, LLM=LLM))
    g.add_node("show_email", show_email)
    g.add_node("call_send_email", call_send_email)

    # Add edges to the graphs
    g.add_edge(START, "gather_information")
    g.add_edge("gather_information", "validate_information")
    g.add_conditional_edges(
        "validate_information",
        validate_information_edge,
        {
            "complete": "draft_email",
            "incomplete": "gather_information",
        },
    )
    g.add_edge("draft_email", "show_email")
    g.add_conditional_edges(
        "show_email",
        show_email_edge,
        {
            "send": "call_send_email",
            "update": "draft_email",
        },
    )
    g.add_edge("call_send_email", END)

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    run_loop(lg_graph)


if __name__ == "__main__":
    main()
