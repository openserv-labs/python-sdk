from src import Agent, Capability
from src import AgentOptions
from pydantic import BaseModel
from typing import Dict, Any, List
import os
from pathlib import Path

# Define argument models
class GreetArgs(BaseModel):
    name: str

class FarewellArgs(BaseModel):
    name: str

class HelpArgs(BaseModel):  # Define an empty schema for the help command
    pass

# Define async functions for capabilities
async def greet_run(data, messages):
    return f"Hello, {data.name}! How can I help you today?"

async def farewell_run(data, messages):
    return f"Goodbye, {data.name}! Have a great day!"

async def help_run(data, messages):
    return "Available commands: greet, farewell, help"

# Initialize the agent
def create_agent() -> Agent:
    """Create and configure the marketing agent."""
    system_prompt_path = Path(__file__).parent.joinpath('system.md')
    if not system_prompt_path.exists():
        raise FileNotFoundError("system.md not found in examples directory")

    agent = Agent(
        AgentOptions(
            system_prompt=system_prompt_path.read_text(),
            api_key=os.getenv('OPENSERV_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    )

    # Create and add capabilities
    greet_capability = Capability(
        name="greet",
        description="Greet a user by name",
        schema=GreetArgs,
        run=greet_run
    )

    farewell_capability = Capability(
        name="farewell",
        description="Say goodbye to a user",
        schema=FarewellArgs,
        run=farewell_run
    )

    help_capability = Capability(
        name="help",
        description="Show available commands",
        schema=HelpArgs,
        run=help_run
    )

    # Add capabilities to agent
    agent.add_capabilities([
        greet_capability,
        farewell_capability,
        help_capability
    ])

    return agent

if __name__ == '__main__':
    agent = create_agent()
    agent.start()
