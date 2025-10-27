from datetime import datetime
from functools import partial
from operator import add
import os
from pprint import pprint
from typing import Annotated, Literal, Optional, Sequence

from langchain_community.embeddings.llamacpp import LlamaCppEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
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

    question_type: Literal["internal", "search"]


class UserSatisfaction(BaseModel):
    """
    Pydantic Model for the user's satisfaction with the result.
    """

    user_satisfaction: Literal["satisfied", "unsatisfied"]


class SearchToolRequest(BaseModel):
    """
    Pydantic Model for the search tool.
    """

    queries: list[str]


class RetrieveToolRequest(BaseModel):
    """
    Pydantic Model for the retrieve tool.
    """

    query: str


class SynthesizedResults(BaseModel):
    """
    Pydantic Model for the final synthesized result.
    """

    synthesized_result: str


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

    # The results from the retrieve tool.
    retrieve_results: Annotated[list, add] = []

    # The queries to the search tool.
    search_queries: Annotated[list[str], add] = []

    # The results from the search tool.
    search_results: Annotated[list, add] = []

    # The final synthesized result.
    synthesized_result: Optional[str] = None


def determine_question_type(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are a helpful question router in place to route the user's prompt to the appropriate tool.
    You are given a user prompt and you need to route the prompt to the appropriate tool.
    If the prompt is about internal company information, you should route the prompt to the internal tool.
    If the prompt is about external information (not about the company), you should route the prompt to the search tool.
    For context, the company is a fence company that sells fences to customers.
    Answer with "internal" or "search" ONLY.
    """
    )

    llm_structured_output = LLM.with_structured_output(QuestionType)
    question_type_result = llm_structured_output.invoke(
        [system_message, *state.messages]
    )
    return {"question_type": question_type_result}


def question_type_edge(state: GraphState):
    return state.question_type.question_type


def retrieve_tool(state: GraphState, LLM, vs: InMemoryVectorStore) -> dict:
    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to retrieve the results from the retrieve tool.
        You are given a user prompt and you need to retrieve the results from the retrieve tool.
        The user prompt is: {state.messages[-1].content}
        """
    )

    llm_structured_output = LLM.with_structured_output(RetrieveToolRequest)
    retrieve_request_result = llm_structured_output.invoke(
        [system_message, *state.messages]
    )

    retrieve_results = vs.similarity_search_with_score(
        retrieve_request_result.query, k=3
    )

    retrieve_results = [
        {"content": doc.page_content, "metadata": doc.metadata, "score": score}
        for doc, score in retrieve_results
    ]

    return {"retrieve_results": retrieve_results}


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


def synthesize_results(state: GraphState, LLM) -> dict:
    system_message = SystemMessage(
        content=f"""You are an assistant whose task is to *synthesize* results from the retrieved and searched data.  
        
        Use only the provided tool outputs:  
        - Retrieve results: {state.retrieve_results if state.retrieve_results else "None"}  
        - Search results: {state.search_results if state.search_results else "None"}  
        
        **Citation rules:**  
        - Quote exact text from the sources when stating facts.  
        - Follow each quote with `(Source: filename.txt, chunk_id: 1)`.  
        - Do not make any claim without supporting evidence; if evidence is absent, say: “Insufficient evidence to answer.” followed by potential clarifications to the user.
        
        Format your response with inline citations for every fact.  
        """
    )

    llm_structured_output = LLM.with_structured_output(SynthesizedResults)
    synthesized_result_result = llm_structured_output.invoke(
        [system_message, *state.messages]
    )

    messages = [
        *state.messages,
        AIMessage(
            content=f"Synthesized result: {synthesized_result_result.synthesized_result}"
        ),
    ]

    return {
        "messages": messages,
        "synthesized_result": synthesized_result_result.synthesized_result,
    }


def check_user_satisfaction(state: GraphState, LLM) -> dict:
    """
    A node to check if the user is satisfied with the result.
    Waits for user input to confirm satisfaction or dissatisfaction.
    """

    feedback = interrupt("Are you satisfied with the result?")

    system_message = SystemMessage(
        content=f"""You are a helpful assistant in place to check if the user is satisfied with the result.
        If the user asks ANY following questions, you should return "unsatisfied".
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


def initialize_rag():
    emb = LlamaCppEmbeddings(
        model_path="src/models/all-MiniLM-L6-v2-Q5_K_M.gguf",
        n_ctx=512,
        n_batch=8,
        device="mps",
        verbose=False,
    )

    vs = InMemoryVectorStore(embedding=emb)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )

    # read file + create Document
    for filename in os.listdir("src/docs"):
        if filename.startswith("."):
            continue

        with open(f"src/docs/{filename}", "r", encoding="utf-8") as f:
            text = f.read()
            chunks = text_splitter.split_text(text)
            docs = [
                Document(
                    page_content=chunk, metadata={"source": filename, "chunk_id": i}
                )
                for i, chunk in enumerate(chunks)
            ]
            vs.add_documents(docs)

    return vs


def main():
    LLM = get_lc_llm()
    checkpoint_saver = InMemorySaver()

    vs = initialize_rag()

    # GraphState is a pydantic model that defines the state of the graph
    g = StateGraph(GraphState)

    # Add nodes to the graph
    g.add_node("determine_question_type", partial(determine_question_type, LLM=LLM))
    g.add_node("check_user_satisfaction", partial(check_user_satisfaction, LLM=LLM))
    g.add_node("retrieve_tool", partial(retrieve_tool, LLM=LLM, vs=vs))
    g.add_node("search_tool", partial(search_tool, LLM=LLM))
    g.add_node("synthesize_results", partial(synthesize_results, LLM=LLM))

    # Add edges to the graphs
    g.add_edge(START, "determine_question_type")
    g.add_conditional_edges(
        "determine_question_type",
        question_type_edge,
        {"internal": "retrieve_tool", "search": "search_tool"},
    )
    g.add_edge("retrieve_tool", "synthesize_results")
    g.add_edge("search_tool", "synthesize_results")
    g.add_edge("synthesize_results", "check_user_satisfaction")
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
