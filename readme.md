AI Agent - PDF Answer Extraction with Agentic RAG

## Overview

This project implements an AI agent that leverages the capabilities of OpenAI's GPT model and agentic Retrieval-Augmented Generation (RAG) to extract answers from large PDF documents. The agent processes questions provided by the user, searches through the document content, and returns the corresponding answers in a structured JSON format. The answers can then be posted to Slack for easy sharing.

![diagram](https://github.com/user-attachments/assets/9067f1aa-8c24-4bbf-bcdb-e1b82efa3da9)

## Problem Statement

The AI agent is designed to handle the following tasks:

1. Extract relevant answers from a large PDF document using a retrieval-based approach.
2. Provide answers to a predefined set of questions or user-generated queries.
3. Post the results in a structured JSON format to Slack.

The solution avoids using pre-built chains from frameworks like LangChain Instead, it implements custom logic to ensure accuracy, flexibility, and modularity.

## Agentic RAG

This implementation leverages agentic RAG, where the agent uses the retrieval system to fetch relevant context for answering questions. The agent retries with semantically different queries if the first attempt doesn't provide sufficient information. It ensures the answer is complete and relevant by calling the retriever multiple times if necessary.

**Agent Workflow:**

1. Receive a question from the user.
2. Query the retriever multiple times with different variations of the question.
3. Return a concise, complete answer.

Example:

```python
def run_agentic_rag(question: str) -> str:
    enhanced_question = f"""
    Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
    give a comprehensive answer to the question below.

    Question:
    {question}
    """
    return agent.run(enhanced_question)
```

## Features

- Extract answers from a PDF document based on user questions.
- Supports the GPT-4o-mini model for processing.
- Agentic RAG ensures comprehensive answer retrieval through multiple semantic queries.
- Provides structured JSON output with the question and answer pair.
- Returns "Data Not Available" for low-confidence answers.

## Example Questions & Answers

Here are some example questions and their respective answers extracted from the PDF:

```json
{
  "question": "What is the name of the company?",
  "answer": "xyz, Inc."
}
```

Demo

## Video Demo

Watch the video demo of the AI Agent in action:

## Video Demo

https://github.com/user-attachments/assets/e22958bd-1f19-434e-a550-ddf85bda7847

