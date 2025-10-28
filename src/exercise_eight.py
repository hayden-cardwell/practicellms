from datetime import datetime
from functools import partial
from operator import add
import os
from pprint import pprint
from turtle import st
from typing import Annotated, Literal, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from shared.lc_llm import get_lc_llm


class CustomerInformation(BaseModel):
    """
    Pydantic Model defining the customer information.
    """

    full_name: str
    property_address: str
    phone_number: str
    email_address: str


class ProjectDetails(BaseModel):
    """
    Pydantic Model defining the project details.
    """

    property_type: Literal["residential", "commercial"]
    fence_type: Literal[
        "privacy", "decorative", "security", "pool", "chain link", "commercial"
    ]
    total_linear_footage: Annotated[float, Field(ge=0)]
    fence_height_in_feet: Literal[4, 5, 6, 8]
    preferred_material: Literal[
        "Cedar", "Pine", "Composite", "Aluminum", "Steel", "Vinyl"
    ]
    location_on_property: Literal["backyard", "front yard", "perimeter", "other"]


class GateRequirements(BaseModel):
    """
    Pydantic Model defining the gate requirements.
    """

    number_of_gates: Annotated[int, Field(ge=0)]
    gate_type: Literal["single", "double", "none"]
    gate_width_in_feet: Optional[Annotated[float, Field(ge=0)]] = None
    automation_requirements: bool = False


class AdditionalDetails(BaseModel):
    """
    Pydantic Model defining the additional details.
    """

    special_requirements: Optional[str] = None
    preferred_timeline: Optional[str] = None
    obstacles: Optional[str] = None
    special_site_conditions: Optional[str] = None


class QuoteInformation(BaseModel):
    """
    Pydantic Model defining the information needed to generate a quote.
    """

    customer_information: CustomerInformation
    project_details: ProjectDetails
    gate_requirements: GateRequirements
    additional_details: AdditionalDetails


class GraphState(BaseModel):
    """
    Pydantic Model defining the state for the graph.
    """

    # Messages are added to the state by the add_messages function.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    quote_information: Optional[QuoteInformation] = None


# TOOLS
def validate_tool(state: GraphState) -> dict:
    """
    Tool to determine if all information has been gathered. Deterministic and not using the LLM.
    """

    pass


# NODES
def quote_generator(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are a professional information gatherer for Superior Fence Solutions, a licensed and insured fencing company established in 2018 with a BBB A+ rating.

        Your role is to gather ALL required information from the customer before a quote can be generated. You must collect the following details through conversation:

        REQUIRED INFORMATION:
        1. Customer Information:
        - Full name
        - Property address
        - Phone number
        - Email address

        2. Project Details:
        - Property type (residential or commercial)
        - Fence type (privacy, decorative, security, pool, chain link, etc.)
        - Total linear footage needed (or dimensions to calculate)
        - Fence height (4ft, 5ft, 6ft, or 8ft)
        - Preferred material (Cedar, Pine, Composite, Aluminum, Steel, Vinyl, etc.)
        - Location on property (backyard, front yard, perimeter, etc.)

        3. Gate Requirements:
        - Number of gates needed
        - Gate type (single, double, or none)
        - Gate width
        - Any automation requirements

        4. Additional Details:
        - Any special requirements or custom features
        - Preferred timeline
        - Any obstacles or special site conditions

        Ask questions one at a time or in small groups to make the conversation natural and not overwhelming. Be friendly and professional. If the customer is unsure about measurements or specifications, offer to schedule a free on-site consultation.

        Available services: Privacy fences (starting at $28/linear foot), decorative fences, security fencing, pool fencing, commercial fencing, gate automation, and custom fabrication.

        Materials available: Wood (Cedar, Pine, Composite), Metal (Aluminum, Steel, Wrought Iron, Chain Link), Vinyl, and specialty materials.

        Service area: Builderville County and surrounding areas (up to 100 miles for large projects).

        Once you have collected ALL required information, confirm the details with the customer before proceeding to quote generation."""
    )

    llm_structured_output = LLM.with_structured_output(QuoteInformation)


def gather_information(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to gather the information from the user.
        You will be given a list of information to gather from the user. 
        You've already asked the user if it's okay, so proceed to ask ONE question at a time to make the conversation natural and not overwhelming. 
        Be friendly and professional. 
        If the customer is unsure about measurements or specifications, offer to schedule a free on-site consultation.
        Start with the customer information of name, address, phone number, and email address."""
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
    pass


def validate_information_edge(state: GraphState) -> Literal["complete", "incomplete"]:
    """Returns 'complete' if all information is gathered, 'incomplete' otherwise."""
    return "complete" if state.quote_information is not None else "incomplete"


def run_loop(lg_graph) -> None:
    # Define the thread ID for the conversation, could be random
    thread = {"configurable": {"thread_id": "1"}}

    # Context always starts with pre-defined AI message
    command = {
        "messages": AIMessage(
            content="""Hello, I'm an AI assistant working for Superior Fence Solutions. 
            I understand you would like a quote on a fence or gate project. 
            I will ask you a few questions to gather the information needed to generate a quote, if that's okay with you."
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
    g.add_node("quote_generator", partial(quote_generator, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "gather_information")
    g.add_edge("gather_information", "validate_information")
    g.add_conditional_edges(
        "validate_information",
        validate_information_edge,
        {"complete": "quote_generator", "incomplete": "gather_information"},
    )
    g.add_edge("quote_generator", END)

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    run_loop(lg_graph)


if __name__ == "__main__":
    main()
