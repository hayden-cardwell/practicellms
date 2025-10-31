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


class CustomerInformation(BaseModel):
    """
    Pydantic Model defining the customer information.
    """

    full_name: str = Field(description="The full name of the customer.")
    property_address: str = Field(
        description="The property address the customer is requesting a quote for."
    )
    phone_number: str = Field(
        description="The phone number the customer can be reached at."
    )
    email_address: str = Field(
        description="The email address the customer can be reached at and which the quote will be sent to."
    )


class ProjectDetails(BaseModel):
    """
    Pydantic Model defining the project details.
    """

    property_type: Literal[
        "residential",
        "commercial",
    ] = Field(
        description="The type of property the customer is requesting a quote for."
    )

    fence_type: Literal[
        "privacy",
        "decorative",
        "security",
        "pool",
        "chain link",
        "commercial",
    ] = Field(description="The type of fence the customer is requesting a quote for.")

    total_linear_footage: Annotated[
        float,
        Field(
            ge=0,
            description="The total linear footage of the fence the customer is requesting a quote for.",
        ),
    ]
    fence_height_in_feet: Literal[
        4,
        5,
        6,
        8,
    ] = Field(
        description="The height of the fence the customer is requesting a quote for."
    )

    preferred_material: Literal[
        "Cedar",
        "Pine",
        "Composite",
        "Aluminum",
        "Steel",
        "Vinyl",
    ] = Field(
        description="The preferred material for the fence the customer is requesting a quote for."
    )

    location_on_property: Literal[
        "backyard",
        "front yard",
        "perimeter",
        "other",
    ] = Field(
        description="The location on the property the customer is requesting a quote for."
    )


class GateRequirements(BaseModel):
    """
    Pydantic Model defining the gate requirements.
    """

    number_of_gates: Annotated[
        int,
        Field(
            ge=0,
            description="The number of gates the customer is requesting a quote for.",
        ),
    ]

    gate_type: Literal[
        "single",
        "double",
        "none",
    ] = Field(description="The type of gate(s) the customer is requesting a quote for.")

    gate_width_in_feet: Optional[
        Annotated[
            float,
            Field(
                ge=0,
                description="The width of the gate(s) the customer is requesting a quote for.",
            ),
        ]
    ] = None

    automation_requirements: bool = Field(
        description="Whether the customer is requesting a quote for automated gates."
    )


class AdditionalDetails(BaseModel):
    """
    Pydantic Model defining the additional details.
    """

    special_requirements: Optional[str] = Field(
        description="Any special requirements or custom features the customer is requesting a quote for. Note that custom features are subject to further review for feasibility."
    )
    preferred_timeline: Optional[str] = Field(
        description="The preferred timeline, including any deadlines or desired completion dates, for the project the customer is requesting a quote for."
    )
    obstacles: Optional[str] = Field(
        description="Any physical obstacles (e.g., trees, rocks, slopes, existing structures) that may affect the fence installation."
    )
    special_site_conditions: Optional[str] = Field(
        description="Any unique site conditions (e.g., soil type, drainage issues, underground utilities, access limitations) that may impact the project."
    )


class QuoteInformation(BaseModel):
    """
    Pydantic Model defining the information needed to generate a quote.
    """

    customer_information: Optional[CustomerInformation] = None
    project_details: Optional[ProjectDetails] = None
    gate_requirements: Optional[GateRequirements] = None
    additional_details: Optional[AdditionalDetails] = None


class GraphState(BaseModel):
    """
    Pydantic Model defining the state for the graph.
    """

    # Messages are added to the state by the add_messages function.
    messages: Annotated[Sequence[BaseMessage], add_messages]
    quote_information: Optional[QuoteInformation] = None
    collected_fields: dict = {}


class InformationCollector:
    def __init__(self, root_model: type[BaseModel]):
        self.root_model = root_model
        self.schema = self.extract_schema()
        self.priority_order = [
            "CustomerInformation",
            "ProjectDetails",
            "GateRequirements",
            "AdditionalDetails",
        ]

    def extract_schema(self) -> dict:
        """
        Extract the schema from the root model.
        """
        return self.root_model.model_json_schema()

    def get_next_field(self, collected_fields: dict):
        """
        returns the next field to collect based on the collected fields.
        """

        for sub_model_name in self.priority_order:
            if sub_model_name in self.schema["$defs"]:
                sub_model = self.schema["$defs"][sub_model_name]
            else:
                continue

            fields_in_sub_model = [p for p in sub_model["properties"].keys()]
            for field_key in fields_in_sub_model:
                if field_key not in collected_fields:

                    # Special handling for any_of fields
                    if "anyOf" in sub_model["properties"][field_key]:
                        valid_types = sub_model["properties"][field_key]["anyOf"]
                    else:
                        valid_types = [sub_model["properties"][field_key]["type"]]

                    return_dict = {
                        "field_key": field_key,
                        "field_pretty_name": sub_model["properties"][field_key][
                            "title"
                        ],
                        "field_description": sub_model["properties"][field_key][
                            "description"
                        ],
                        "field_type": valid_types,
                        "required": field_key in sub_model["required"],
                    }
                    pprint(return_dict)
                    print("-" * 80)
                    return return_dict
            return None


# TOOLS
def validate_tool(state: GraphState):
    """
    Tool to determine if all information has been gathered. Deterministic and not using the LLM.
    """

    if state.quote_information is None:
        return {}
    return state.quote_information.model_dump()


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


def gather_information(state: GraphState, LLM, ic: InformationCollector) -> dict:
    # TODO: Fix issue where the same field is asked for multiple times bc collected_fields is never appended to
    """Node for gathering information from the user.
    Will be looped back to, one field at a time, until all information is gathered.
    Because of this, we need to create it to be stateless, so it's field agnostic."""

    next_field = ic.get_next_field(state.collected_fields)
    if next_field is None:
        return {}

    # next_field should look like:
    """
    {"field_name": field_name,
    field_description: field_description,
    field_type: field_type,
    required: required, }
    """

    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to gather the information from the user.
        You will be given a list of information to gather from the user. 
        Be friendly and professional. 
        If the customer is unsure about measurements or specifications, offer to schedule a free on-site consultation.
        Ask ONE question at a time to make the conversation natural and not overwhelming.
        The piece of information you're currently attempting to collect is: {next_field}"""
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


def stash_information(state: GraphState, LLM, ic: InformationCollector) -> dict:
    """Node for stashing information in the state."""

    next_field = ic.get_next_field(state.collected_fields)
    if next_field is None:
        return {}

    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to validate information gathered from the user.
        The piece of information you're currently attempting to collect is: {next_field}"""
    )

    # The structured output is the next field's type
    # TODO: Implement validation type checking somehow, IDK how.
    llm_structured_output = LLM.with_structured_output(next_field["field_type"])
    stashed_information = llm_structured_output.invoke(
        [system_message, *state.messages]
    )

    return {
        "collected_fields": {**state.collected_fields},
    }


def stash_information_edge(state: GraphState) -> Literal["complete", "incomplete"]:
    """Returns 'complete' if all information is gathered, 'incomplete' otherwise."""
    return "complete" if state.collected_fields is not None else "incomplete"
    pass


def run_loop(lg_graph) -> None:
    # Define the thread ID for the conversation, could be random
    thread = {"configurable": {"thread_id": "1"}}

    # Context always starts with pre-defined AI message
    command = {
        "messages": AIMessage(
            content="""Hello, I'm an AI assistant working for Superior Fence Solutions. 
            I understand you would like a quote on a fence or gate project. 
            I will ask you a series of questions to gather the information needed to generate a quote.
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
    ic = InformationCollector(QuoteInformation)
    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("gather_information", partial(gather_information, LLM=LLM, ic=ic))
    g.add_node("stash_information", partial(stash_information, LLM=LLM, ic=ic))
    g.add_node("quote_generator", partial(quote_generator, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "gather_information")
    g.add_edge("gather_information", "stash_information")
    g.add_conditional_edges(
        "stash_information",
        stash_information_edge,
        {"complete": "quote_generator", "incomplete": "gather_information"},
    )
    g.add_edge("stash_information", "quote_generator")
    g.add_edge("quote_generator", END)

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    run_loop(lg_graph)


if __name__ == "__main__":
    main()
