from functools import partial
import os
from pprint import pprint
from typing import Annotated, Literal, Optional, Sequence

from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
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

    # The user's satisfaction with the result.
    user_satisfaction: Optional[UserSatisfaction] = None


def determine_question_type(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to answer the user's prompt.
    You are given a user prompt and you need to answer the prompt.
    Answer with a concise and helpful response.
    """
    )

    llm_structured_output = LLM.with_structured_output(QuestionType)
    question_type_result = llm_structured_output.invoke(
        [system_message, *state.messages]
    )
    return {"question_type": question_type_result}


def question_type_edge(state: GraphState):
    return state.question_type.question_type


def check_user_satisfaction(state: GraphState, LLM) -> dict:
    """
    A node to check if the user is satisfied with the result.
    Waits for user input to confirm satisfaction or dissatisfaction.
    """

    feedback = interrupt("Are you satisfied with the result?")

    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to check if the user is satisfied with the result.
        If the user asks ANY following questions, you should return "unsatisfied"
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
        print("Using default query of 'What is the highest privacy fence offered?'")
        user_input = "What is the highest privacy fence offered?"

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
            break


def main():
    LLM = get_lc_llm()
    checkpoint_saver = InMemorySaver()

    emb = LlamaCppEmbeddings(
        model_path="src/models/all-MiniLM-L6-v2-Q5_K_M.gguf",
        n_ctx=2048,
        device="mps",
        verbose=False,
    )

    vs = InMemoryVectorStore(embedding=emb)

    # read file + create Document
    for f in os.listdir("src/docs"):
        with open(f"src/docs/{f}", "r", encoding="utf-8") as f:
            text = f.read()
            doc = Document(page_content=text, metadata={"source": f})
            vs.add_documents([doc])

    results = vs.similarity_search_with_score("Customer quote", k=3)
    pprint(f"Results: {results}")

    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("determine_question_type", partial(determine_question_type, LLM=LLM))
    g.add_node("check_user_satisfaction", partial(check_user_satisfaction, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "determine_question_type")
    g.add_conditional_edges(
        "determine_question_type",
        question_type_edge,
        {"other": "check_user_satisfaction"},
    )
    g.add_edge("check_user_satisfaction", END)

    lg_graph = g.compile(checkpointer=checkpoint_saver)

    png_data = lg_graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    # run_loop(lg_graph)


if __name__ == "__main__":
    main()
