"""
LangChain agent with local Ollama LLM for answering energy consumption questions.
Uses modern LangChain APIs (no deprecated methods).
Queries real training data from CSV when needed.
"""

import json
import logging
from datetime import datetime

import pandas as pd
from langchain.agents import AgentExecutor, create_react_agent
from langchain.hub import pull
from langchain.tools import Tool
from langchain_community.llms import Ollama

from src.constants import APPLIANCES_CSV, TARGET_COLUMN

# Configure logging
logger = logging.getLogger(__name__)


# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:120b-cloud"


def _load_energy_data() -> dict:
    """Load energy consumption statistics from training data CSV."""
    try:
        if not APPLIANCES_CSV.exists():
            logger.warning(f"Energy data file not found at {APPLIANCES_CSV}")
            return {}

        df = pd.read_csv(APPLIANCES_CSV)
        if TARGET_COLUMN not in df.columns:
            return {}

        consumption = df[TARGET_COLUMN].astype(float)
        return {
            "avg_consumption": f"{consumption.mean():.2f} Wh",
            "peak_consumption": f"{consumption.max():.2f} Wh",
            "min_consumption": f"{consumption.min():.2f} Wh",
            "std_deviation": f"{consumption.std():.2f} Wh",
        }
    except Exception as e:
        logger.error(f"Failed to load energy data: {e}")
        return {}


def setup_llm_agent() -> AgentExecutor:
    """
    Initialize LangChain REACT agent with local Ollama LLM.

    Uses modern LangChain APIs (no deprecated methods).
    Connects to Ollama service running on localhost:11434.

    Returns:
        AgentExecutor instance

    Raises:
        ConnectionError: If Ollama service is not available
    """
    logger.info(f"Initializing LLM agent with Ollama ({OLLAMA_MODEL})...")

    try:
        llm = Ollama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            top_p=0.9,
            timeout=30.0,
        )
        logger.info(f"Connected to Ollama at {OLLAMA_BASE_URL}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama at {OLLAMA_BASE_URL}: {e}")
        raise ConnectionError(
            f"Could not connect to Ollama LLM service at {OLLAMA_BASE_URL}. " f"Ensure Ollama is running: ollama serve"
        ) from e

    # Define tools
    def query_energy_data(query: str) -> str:
        """Query historical energy consumption statistics from training data."""
        data = _load_energy_data()
        if not data:
            return "No energy data available. Please check the data file."
        return json.dumps(data)

    tools = [
        Tool(
            name="QueryEnergyData",
            func=query_energy_data,
            description="Get historical energy consumption statistics (average, peak, min, std dev)",
        ),
    ]

    # Load REACT prompt template
    prompt = pull("hwchase17/react")

    # Create agent using modern API
    agent = create_react_agent(llm, tools, prompt)

    # Wrap in AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=20,
        handle_parsing_errors=True,
    )

    logger.info("LLM agent initialized successfully")
    return agent_executor


def generate_explanation(agent: AgentExecutor, forecast_value: float, question: str) -> str:
    """
    Generate natural-language explanation using the Ollama LLM agent.

    Args:
        agent: AgentExecutor instance
        forecast_value: Current or predicted energy load in Wh
        question: User's question about energy consumption

    Returns:
        Natural-language explanation from the LLM

    Raises:
        RuntimeError: If the agent fails to generate an explanation
    """
    context = (
        f"Current energy forecast: {forecast_value:.2f} Wh\n"
        f"Time: {datetime.now().strftime('%H:%M')}\n"
        f"Question: {question}"
    )

    try:
        logger.debug(f"Generating explanation for: {question}")
        result = agent.invoke({"input": context})
        response: str = result.get("output", "")
        logger.debug(f"Agent response: {response}")
        return response
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise RuntimeError(f"Failed to generate explanation: {str(e)}") from e
