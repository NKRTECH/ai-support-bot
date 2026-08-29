# SmartTech Customer Support System

An intelligent, multi-agent customer support system for SmartTech Electronics. 

## Overview
This repository contains the core AI support agent infrastructure for SmartTech. It handles customer inquiries, processes refunds, checks order statuses, and provides technical troubleshooting through a Retrieval-Augmented Generation (RAG) pipeline based on company policies.

## Key Features
- **Intelligent Triage:** Automatically classifies user intent to route to the correct specialized sub-agent.
- **RAG Knowledge Base:** Hybrid search over policy documents for accurate, grounded answers.
- **Agentic Actions:** Secure tool-calling for order lookups and refunds with Human-in-the-Loop (HITL) approval gates.
- **Interoperable Ecosystem:** Built on the Agent-to-Agent (A2A) protocol, exposing agents as discoverable HTTP services.
- **Model Context Protocol (MCP):** Connects to external databases and internal APIs via stateless HTTP MCP servers.

## Setup & Installation

1. **Environment Setup**
   Ensure you have Python 3.11+ installed.
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Configuration**
   Copy the example environment file and configure your API keys:
   ```bash
   cp .env.example .env
   ```
   *Required: `GEMINI_API_KEY` (Available via [Google AI Studio](https://aistudio.google.com))*

3. **Running the Application**
   Start the interactive terminal client:
   ```bash
   python app.py
   ```
