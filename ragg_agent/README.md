# LangChain & Gemini AI Agent Experiments (`ragg_agent`)

A collection of Python scripts demonstrating LangChain workflows, parallel runnables, message handling, and custom tool calling powered by Google Gemini (`gemini-3.6-flash`).

## Project Structure

```text
ragg_agent/
├── Agentt.py             # Basic LangChain agent implementation
├── from_message.py       # Handling LangChain message types (System, Human, AI)
├── new_summ.py           # Text summarization workflows
├── own.py                # Custom agent/chain experiments
├── parallelrunnable.py   # Executing chains concurrently using RunnableParallel
├── requirements.txt      # Project dependencies
├── self_agent.py         # Self-directed / autonomous agent logic
└── toolcalling.py        # Custom tool creation and binding with Gemini