## practiceLLMs — LangChain & LangGraph Practice Exercises

A small, hands-on repo for learning to build AI agents with Python, LangChain, and LangGraph. Inspired by and loosely built off the exercises at [Practice Python](https://www.practicepython.org).

### Why this exists
- Learn the basics fast: prompts, simple tools, graphs, memory, and RAG.
- Keep examples tiny, runnable, and easy to tweak.
- Build a shared vocabulary and a few simple guardrails.

### What’s here
- `langgraph-practice-exercises.md`: 10 short exercises (from “Hello LLM” to testing/observability).
- `src/`: Minimal examples and shared helpers.
  - `src/shared/lc_llm.py`: LLM config via environment variables (model, temperature, base URL, API key).
- `pyproject.toml`: Dependencies and Python version (>= 3.10).

### What you’ll learn
- How to prototype agent behavior quickly (LangChain vs LangGraph).
- Why explicit graphs/state make behavior easier to test and debug.
- Reusable patterns for prompts, tools, retrieval, and simple approval gates.

### Quick start
- Skim `langgraph-practice-exercises.md` and pick an exercise.
- Add a `.env` with your OpenAI settings; run the corresponding script in `src/`.
- Keep changes small; log inputs/outputs so you can see what happened.

### Requirements
- Python 3.10+, LangChain, LangGraph, and an OpenAI-compatible chat model.
- Local `.env` for secrets (don’t commit API keys).

### Credit
- Inspired by the spirit of [Practice Python](https://www.practicepython.org) exercises.

