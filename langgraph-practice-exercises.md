# LangChain & LangGraph Practice Exercises (Practice Python–Style)

Short, focused exercises for learning to build AI agents. Assumes comfort with Python and basic CLI usage. Each exercise includes a concise task, a few “Extras,” and a short “Discussion.”

---

## How to Use
- Pick one exercise and complete the **Task** first.
- Tackle **Extras** to deepen skills.
- Read the **Discussion** for context and key ideas.
- Keep each solution minimal, runnable, and well logged.
- It's highly recommended that you don't use AI to "vibe-code" the solutions. Something like Copilot or Cursor Tab is probably okay, as long as you feel like you're grasping the concepts and not just getting the code.

## Tips

- Keep prompts short and structured.
- Prefer explicit state and small nodes over monolithic calls.
- Log inputs, tool calls, and outputs for every step.
- Add safety caps on loops and tool invocations.
- Version both prompts and agent graphs in your repo.

---

## Exercise 1: Hello, LLM

**Task**
- Load an API key from the environment.
- Send a single user prompt to a chat model.
- Print the model’s reply.

**Extras**
- Ask for the user’s name via `input()` and include it in the prompt.
- Add a `--model` CLI flag to switch models.

**Discussion**
This is the smallest possible LLM loop: prompt → response. Keep I/O simple and avoid hard‑coding secrets. Prefer environment variables or a `.env` loader for local dev.

---

## Exercise 2: A Tiny Prompt Template

**Task**
- Create a **regular text prompt template** (not a chat template) with two variables like `{role}` and `{topic}`.
- These represent placeholders for user inputs — for example:  
  `"As a {role}, write a short explanation about {topic}."`
- Render it with user inputs before calling the model.
- Print both the rendered prompt and the LLM answer.

**Extras**
- Add a temperature parameter and show how it changes outputs.
- Save the rendered prompt and the model reply to a timestamped `.txt` file.

**Discussion**
This exercise uses a simple string-based **prompt template**, not a chat template. You’re just formatting one text prompt and sending it to the model. Later exercises will introduce multi-message chat templates where `{role}` is a structural key (e.g., system vs user vs assistant). For now, focus on templating text variables and understanding how temperature affects response creativity.

---

## Exercise 3: Your First Tool-Using Agent (Calculator)

**Task**
- Build a simple **LangGraph agent (repeating agent loop)** that decides when it needs arithmetic.
- Call a **calculator tool** to do the math.
- Return the final answer to the user.

**Extras**
- Add basic error handling for invalid math such as division by zero.
- Show a transparent tool log like “Calling calculator with `3 * 7`.”

**Discussion**
This exercise uses **LangGraph’s repeating agent loop**, not LangChain Agents. The goal is to understand the implicit agent flow where the model dynamically selects and invokes tools inside the loop. You’ll implement reasoning → tool selection → action → observation → final answer flow.
In the next exercise, you will rebuild this logic with an explicit **StateGraph** to replace the implicit loop with explicit edges.

---

## Exercise 4: Rebuild the Agent in LangGraph

**Task**
- Recreate Exercise 3 using an explicit **StateGraph** (no built‑in repeating loop).
- Define a minimal **StateGraph** with nodes such as `plan`, `act`, and `answer`.
- Add an edge that calls the calculator and stores the result in state.

**Extras**
- Add a stop condition that exits when the agent has an answer.
- Print the final **state** as JSON to show what happened.

**Discussion**
LangGraph gives explicit control over agent state and edges between steps. Do not use the repeating agent loop here; route control flow explicitly via edges and conditions. Start with a tiny graph to learn state, nodes, and edge conditions before expanding.

---

## Exercise 5: Search Tooling (Web Q&A)

**Task**
- Add a **search tool** to your LangGraph agent.
- Decide when the agent should use the web.
- Read top results and synthesize an answer with short citations.

**Extras**
- Add a guardrail: if the query is purely math, skip web search.
- Limit to N results and include a one‑line rationale like “Used web search because the question was time‑sensitive.”

**Discussion**
This is the classic tool choice: calculator vs search. LangGraph lets you encode the decision as edges and conditions that are easy to test.

---

## Exercise 6: Memory (Conversation Summary vs Window)

**Task**
- Extend the LangGraph agent with **memory** for a short multi‑turn chat.
- Implement either **summary memory** or a **rolling window** of the last N messages.
- Correctly recall a user preference from earlier in the session.

**Extras**
- Support both memory modes behind a CLI flag.
- Persist memory to a local file so the session can be resumed.

**Discussion**
Memory is essential for coherent multi‑turn behavior. Keep state minimal but useful. Summaries reduce token usage. Windows keep wording precise.

---

## Exercise 7: Retrieval‑Augmented Generation (Local Docs)

**Task**
- Give your agent a **retriever tool** backed by local docs.
- Chunk 2–3 markdown files and build a simple vector store.
- Expose a `retrieve(query)` tool and decide when to call it vs search vs calculator.

**Extras**
- Require **source attribution**: answers must list which local files were used.
- Experiment with chunk sizes and measure impact on answer quality.

**Discussion**
RAG turns your agent into a private expert. In LangGraph, retrieval is another node the policy can route to. Keep the graph explicit and testable.

---

## Exercise 8: Human‑in‑the‑Loop (Approval Gate)

**Task**
- Add a **human approval** step before performing a “write” action such as saving a file or sending a mock email.
- If the plan includes a write, pause and display a diff or preview.
- Wait for `y/n` input and proceed or abort.

**Extras**
- Add a timeout that defaults to “abort” if there’s no response.
- Log approved actions to a CSV.

**Discussion**
Human‑in‑the‑loop is a core pattern. Implement an interrupt or approval node so risky actions require confirmation.

---

## Exercise 9: Multi‑Turn Planning with Custom State

**Task**
- Implement a two‑phase flow in LangGraph.
- **Plan**: create a bullet list of steps such as “Research”, “Draft”, “Revise”.
- **Execute**: iterate over steps, using tools as needed.
- Track progress with custom state like `{ plan: [...], current_step, artifacts: [...] }`.

**Extras**
- Enable **retry** of a failed step with a different tool.
- Add a `--max-steps` safety cap and return a partial result if exceeded.

**Discussion**
Custom state makes the plan explicit, debuggable, and resumable. This highlights LangGraph’s strength: controllable node flow and termination.

---

## Exercise 10: Observability, Tests, and “Done” Criteria

**Task**
- Add **structured outputs** such as a Pydantic schema for final answers.
- Write **unit tests** for at least two tools and one end‑to‑end path.
- Log or trace node transitions and decisions.
- Implement a **done criteria** function that checks if requirements were met, such as minimum answer length and at least one citation when web was used.

**Extras**
- Add an `--eval` mode that runs a small suite of prompts and prints pass or fail.
- Emit a JSON report like `/reports/run-<timestamp>.json` with timings and decisions.

**Discussion**
Agents are software. Treat tools and graphs as units to assert on. Use tests, logs, and explicit acceptance criteria to keep behavior predictable.

---

## Suggested Progression

1) Exercises 1–2: Core prompting and templating.  
2) Exercises 3–4: Tools and explicit graph control.  
3) Exercises 5–7: Information access via search and retrieval.  
4) Exercise 8: Human oversight.  
5) Exercises 9–10: Planning, reliability, and evaluation.

---

