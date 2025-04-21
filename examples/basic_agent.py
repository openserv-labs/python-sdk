"""
Basic agent implementation for the OpenServ Python SDK.

This example demonstrates the core components of the OpenServ SDK:

1. Agent: The main class that handles communication with the OpenServ platform
2. Capability: Reusable functions that your agent can perform
3. Schema: Type definitions for your capabilities' inputs using Pydantic
4. Message Handling: How to process and respond to user messages
5. System Prompt: Provides context for the LLM that define the agent's personality and behavior

This agent implements three simple capabilities to demonstrate these concepts.
"""

from src import Agent, Capability
from src import AgentOptions
from pydantic import BaseModel
import os
import logging
from src.types import RespondChatMessageAction
from dotenv import load_dotenv

# Load environment variables stored in .env file
load_dotenv()

# Configure Python's logging for debugging and monitoring
logger = logging.getLogger(__name__)

# Load system prompt that defines the agent's personality and behavior and provides context for the LLM
with open("examples/system_basic_agent.md", "r") as f:
    system_prompt = f.read()

# Define schemas using Pydantic for automatic validation and type checking
# Schemas define the expected input structure for each capability
class GreetArgs(BaseModel):
    name: str

class FarewellArgs(BaseModel):
    name: str

class HelpArgs(BaseModel):
    pass

def create_agent() -> Agent:
    """Create and configure the agent
    
    1. Initialize an agent with configuration
    2. Add capabilities (tools) to the agent
    3. Set up the agent for handling messages
    
    The agent is the core component that:
    - Manages communication with the OpenServ platform
    - Handles incoming messages
    - Executes capabilities
    - Sends responses back to users
    """

    agent = Agent(
        AgentOptions(
            system_prompt=system_prompt,
            api_key=os.getenv('OPENSERV_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    )

    # Add capabilities 
    # Capabilities are reusable functions your agent can perform
    # Each capability has:
    # - name: Unique identifier
    # - description: What the capability does
    # - schema: Input validation rules
    # - run: The function that implements the capability
    agent.add_capabilities([
        # Greet capability
        Capability(
            name="greet",
            description="Greet a user by name",
            schema=GreetArgs,
            run=lambda data, _: f"Hello, {data.name}! How can I help you today?"
        ),
        
        # Farewell capability
        Capability(
            name="farewell",
            description="Say goodbye to a user",
            schema=FarewellArgs,
            run=lambda data, _: f"Goodbye, {data.name}! Have a great day!"
        ),
        
        # Help capability
        Capability(
            name="help",
            description="Show available commands",
            schema=HelpArgs,
            run=lambda _, __: "Available commands: greet, farewell, help"
        )
    ])

    return agent

# Custom agent class with simplified message handling
# This shows how to extend the base Agent class to add custom behavior
class BasicAgent(Agent):
    async def respond_to_chat(self, action: RespondChatMessageAction) -> None:
        """Handle chat messages in a more straightforward way
        
        This method demonstrates how the SDK processes messages:
        1. Receives a message from the user
        2. Determines which capability to use
        3. Extracts necessary information from the message
        4. Executes the appropriate capability
        5. Sends the response back to the user
        
        The RespondChatMessageAction contains:
        - messages: The conversation history
        - me: Information about the agent
        - workspace: Information about the workspace
        """
        if not action.messages:
            return

        last_message = action.messages[-1].message
        response = None

        # Simple command detection
        # This shows how to route messages to different capabilities
        if 'greet' in last_message.lower():
            name = extract_name(last_message)
            greet_tool = next((t for t in self.tools if t.name == "greet"), None)
            if greet_tool:
                response = await greet_tool.run(GreetArgs(name=name), [])
        
        elif 'help' in last_message.lower():
            help_tool = next((t for t in self.tools if t.name == "help"), None)
            if help_tool:
                response = await help_tool.run(HelpArgs(), [])
        
        elif any(word in last_message.lower() for word in ['goodbye', 'bye', 'farewell']):
            name = extract_name(last_message)
            farewell_tool = next((t for t in self.tools if t.name == "farewell"), None)
            if farewell_tool:
                response = await farewell_tool.run(FarewellArgs(name=name), [])

        # Default response if no command is detected
        if not response:
            response = "I'm a basic agent that can greet you, say goodbye, or provide help. Try asking for one of these!"

        # Send response back to the user
        # The SDK handles the HTTP communication with the OpenServ platform
        if action.me and action.workspace:
            await self.api_client.post(
                f"/workspaces/{action.workspace.id}/agent-chat/{action.me.id}/message",
                {"message": response}
            )

def extract_name(message: str) -> str:
    """Helper function to extract name from message
    
    This demonstrates how to:
    1. Parse user input
    2. Extract structured data from natural language
    3. Handle different input formats
    
    The SDK doesn't handle this directly - it's up to you to implement
    the logic for extracting information from user messages.
    """
    message = message.lower()
    
    if "i am " in message:
        return message.split("i am ")[1].split(',')[0].split('.')[0].strip().capitalize()
    elif "my name is " in message:
        return message.split("my name is ")[1].split(',')[0].split('.')[0].strip().capitalize()
    
    # Look for names in comma-separated parts
    parts = message.split(',')
    for part in parts:
        clean_part = part.strip()
        if (clean_part and not clean_part in ["i", "me", "greet", "hello", "hi", "farewell", "goodbye", "bye"]
                and len(clean_part) > 2):
            return clean_part.capitalize()
    
    return "there"

if __name__ == '__main__':
    # Create and start the agent
    # The start() method:
    # 1. Initializes the HTTP server
    # 2. Sets up routes for handling messages
    # 3. Starts listening for incoming requests
    agent = create_agent()
    agent.start()
