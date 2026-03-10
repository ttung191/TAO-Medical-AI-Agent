# TAO: Tiered Agentic Oversight for Medical AI

## Introduction
This repository contains the implementation of the TAO (Tiered Agentic Oversight) framework, a hierarchical multi-agent system designed to enhance AI safety, accuracy, and accountability in healthcare diagnostic settings. 

Current Large Language Models (LLMs) pose significant safety risks in clinical environments due to hallucinations and the lack of robust validation mechanisms. TAO addresses these critical vulnerabilities by employing a layered supervision architecture inspired by real-world clinical workflows (e.g., Triage Nurse -> Specialist -> Consultant).

## Key Technical Features
* **Dynamic Agent Recruitment:** The system automatically analyzes patient input to recruit the most relevant medical specialists for the specific case.
* **Strict Rejection Loop (Fail-Fast Mechanism):** Upper-tier agents are governed by strict business rules that enforce the capability to reject and return cases if crucial clinical information (e.g., vital signs, specific medical history, ECG results) is missing. This prevents premature or hallucinated diagnoses.
* **Hierarchical Oversight Workflow:** Strategically allocates tasks across tiers, prioritizing strict validation at the initial assessment level to optimize API latency and cost while maintaining critical safety standards.
* **Live Model Comparison (A/B Testing):** Features an integrated Streamlit dashboard allowing real-time benchmarking of multiple LLMs (e.g., Gemini 2.5 Flash vs. Gemini 1.5 Pro). The system tracks and compares escalation decisions, confidence scores, execution time, and token costs.

## System Architecture
The application orchestrates specialized AI agents across three hierarchical tiers:
1. **Tier 1 (Initial Assessment):** Triage, vital screening, and strict data validation. Determines whether to resolve, escalate, or reject based on data sufficiency.
2. **Tier 2 (Specialist):** In-depth diagnosis, treatment planning, and rigorous review of Tier 1 findings.
3. **Tier 3 (Consultant):** Final oversight, consensus building, and critical decision-making for highly complex scenarios.

## Technology Stack
* **Language:** Python 3.11+
* **Framework:** Streamlit (Multi-page Application)
* **AI/LLM Engine:** Google Generative AI (Gemini 2.5 Flash, Gemini 1.5 Pro)
* **Data Processing:** Pandas

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/ttung191/TAO-Medical-AI-Agent.git](https://github.com/ttung191/TAO-Medical-AI-Agent.git)
   cd TAO-Medical-AI-Agent
