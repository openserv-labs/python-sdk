"""
📚 Basic Agent Example for OpenServ Python SDK 📚

A simple educational example showing core OpenServ SDK concepts.
"""

from src import Agent, Capability, AgentOptions 
from pydantic import BaseModel
import os
import logging
import re
from dotenv import load_dotenv

# Configure logging to prevent sensitive information exposure
class SensitiveDataFilter(logging.Filter):
    """Filter to prevent sensitive data from appearing in logs"""
    def filter(self, record):
        if not hasattr(record, 'getMessage'):
            return True
            
        message = record.getMessage()
        
        # Completely suppress task messages logs that contain the system prompt
        if "Task messages:" in message and "'role': 'system'" in message and "'content':" in message:
            return False
            
        # Replace API keys in log messages
        if any(key in message for key in ['api_key', 'apiKey', 'API_KEY']):
            api_key = str(os.getenv('OPENSERV_API_KEY', ''))
            if api_key:
                record.msg = record.msg.replace(api_key, '******')
            
            # Handle dictionary args
            if hasattr(record, 'args') and isinstance(record.args, dict):
                if 'api_key' in record.args:
                    record.args['api_key'] = '******'
                if 'options' in record.args and isinstance(record.args['options'], dict):
                    if 'api_key' in record.args['options']:
                        record.args['options']['api_key'] = '******'
        
        # Replace system_prompt content in log messages
        if 'system_prompt' in message:
            # Handle direct string replacement
            if isinstance(record.msg, str) and 'system_prompt' in record.msg:
                # Use regex to replace system_prompt content
                record.msg = re.sub(r'(\'system_prompt\':\s*)[\'"].*?[\'"]', r'\1"<hidden>"', record.msg)
            
            # Handle dictionary args
            if hasattr(record, 'args') and isinstance(record.args, dict):
                if 'system_prompt' in record.args:
                    record.args['system_prompt'] = '<hidden>'
                if 'options' in record.args and isinstance(record.args['options'], dict):
                    if 'system_prompt' in record.args['options']:
                        record.args['options']['system_prompt'] = '<hidden>'
                        
        return True

# Apply the filter to all loggers
root_logger = logging.getLogger()
root_logger.addFilter(SensitiveDataFilter())
# Also specifically apply to src.agent logger
agent_logger = logging.getLogger('src.agent')
agent_logger.addFilter(SensitiveDataFilter())

# Colorama is used to color the terminal text
try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class DummyColor:
        def __getattr__(self, name):
            return ""
    Fore = Style = DummyColor()

# Load environment variables from .env file
load_dotenv()

# Define Pydantic schemas for capability
class GreetArgs(BaseModel):
    name: str

class FarewellArgs(BaseModel):
    name: str

class HelpArgs(BaseModel):
    pass

# Print functions for terminal feedback for the user to follow what its happening
def print_header(text):
    """Print a formatted header"""
    if HAS_COLOR:
        print(f"\n{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  {text}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}\n")
    else:
        print(f"\n{'=' * 60}")
        print(f"  {text}")
        print(f"{'=' * 60}\n")

def print_step(emoji, text, color=Fore.BLUE):
    """Print a step with emoji and colored text"""
    if HAS_COLOR:
        print(f"{color}{emoji}  {text}{Style.RESET_ALL}")
    else:
        print(f"{emoji}  {text}")

def print_substep(text, color=Fore.YELLOW):
    """Print a substep with indentation"""
    if HAS_COLOR:
        print(f"{color}   ↪ {text}{Style.RESET_ALL}")
    else:
        print(f"   ↪ {text}")

def print_success(text):
    """Print a success message"""
    if HAS_COLOR:
        print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")
    else:
        print(f"✅ {text}")

def print_error(text):
    """Print an error message"""
    if HAS_COLOR:
        print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")
    else:
        print(f"❌ {text}")

def print_message(label, text, color=Fore.WHITE):
    """Print a message with label and content"""
    if HAS_COLOR:
        print(f"{Fore.BLUE}{label}: {color}{text}{Style.RESET_ALL}")
    else:
        print(f"{label}: {text}")

# Custom error handler to prevent sensitive information exposure
def error_handler(error, context=None):
    """Handle errors without exposing sensitive information"""
    error_msg = str(error)
    # Remove any potential API keys from error messages
    api_key = os.getenv('OPENSERV_API_KEY', '')
    if api_key and api_key in error_msg:
        error_msg = error_msg.replace(api_key, '******')
    
    print_error(f"Error: {error_msg}")
    if context:
        print_substep(f"Context: {context}")

# Basic agent class that inherits from the Agent class from the OpenServ SDK
class BasicAgent(Agent):
    async def respond_to_chat(self, action):
        """Process and respond to chat messages"""
        if not action.messages:
            return

        # Get the most recent message
        last_message = action.messages[-1].message
        print_header("Incoming User Message")
        print_message("User said", last_message, Fore.WHITE)
        
        # Simple rule-based command detection
        print_header("Message Processing")
        print_step("🔍", "Analyzing message for commands")
        
        response = None
        # Check for greeting words first
        if any(word in last_message.lower() for word in ['greet', 'hi', 'hello', 'hey', 'howdy']):
            print_substep("Detected greeting")
            name = self.extract_name(last_message)
            print_step("👤", f"Extracted name: {name}", Fore.MAGENTA)
            
            greet_tool = next((t for t in self.tools if t.name == "greet"), None)
            if greet_tool:
                print_step("🛠️", "Executing 'greet' capability")
                response = await greet_tool.run(GreetArgs(name=name), [])
                print_success(f"Generated greeting for {name}")
                
        elif 'help' in last_message.lower():
            print_substep("Detected 'help' command")
            help_tool = next((t for t in self.tools if t.name == "help"), None)
            if help_tool:
                print_step("🛠️", "Executing 'help' capability")
                response = await help_tool.run(HelpArgs(), [])
                print_success("Generated help information")
                
        elif any(word in last_message.lower() for word in ['goodbye', 'bye', 'farewell']):
            print_substep("Detected 'farewell' command")
            name = self.extract_name(last_message)
            print_step("👤", f"Extracted name: {name}", Fore.MAGENTA)
            
            farewell_tool = next((t for t in self.tools if t.name == "farewell"), None)
            if farewell_tool:
                print_step("🛠️", "Executing 'farewell' capability")
                response = await farewell_tool.run(FarewellArgs(name=name), [])
                print_success(f"Generated farewell for {name}")

        # Default response if no command detected
        if not response:
            print_error("No command detected in message")
            response = "❓ I'm not sure what you're asking.\n\nI'm a basic agent that can:\n• Greet you when you say 'hi' or 'hello'\n• Provide 'help' with available commands\n• Say goodbye when you say 'bye' or 'goodbye'\n\nTry one of these commands!"
            print_step("📝", "Using default response", Fore.YELLOW)

        # Send response back to user
        print_header("Sending Response")
        print_message("Agent response", response, Fore.GREEN)
        
        if action.me and action.workspace:
            print_step("📤", "Posting message to OpenServ platform")
            await self.api_client.post(
                f"/workspaces/{action.workspace.id}/agent-chat/{action.me.id}/message",
                {"message": response}
            )
            print_success("Response sent successfully")

    def extract_name(self, message: str) -> str:
        """Extract name from user message"""
        message = message.lower()
        
        # Common patterns for name extraction
        name_patterns = [
            # "I am [name]" pattern
            (message.split("i am ")[1].split(',')[0].split('.')[0].strip().capitalize(), "Found name after 'I am'") 
            if "i am " in message else None,
            
            # "My name is [name]" pattern
            (message.split("my name is ")[1].split(',')[0].split('.')[0].strip().capitalize(), "Found name after 'my name is'") 
            if "my name is " in message else None,
            
            # "I'm [name]" pattern
            (message.split("i'm ")[1].split(',')[0].split('.')[0].strip().capitalize(), "Found name after 'I'm'") 
            if "i'm " in message else None,
            
            # "This is [name]" pattern
            (message.split("this is ")[1].split(',')[0].split('.')[0].strip().capitalize(), "Found name after 'this is'") 
            if "this is " in message else None,
        ]
        
        # Try the patterns first
        for pattern_result in name_patterns:
            if pattern_result:
                name, explanation = pattern_result
                print_substep(f"{explanation}: {name}")
                return name
        
        # Look for names in comma-separated parts
        parts = message.split(',')
        for part in parts:
            clean_part = part.strip()
            if (clean_part and not clean_part in ["i", "me", "you", "greet", "hello", "hi", "hey", "there", "farewell", "goodbye", "bye"]
                    and len(clean_part) > 2):
                print_substep(f"Extracted potential name from message: {clean_part.capitalize()}")
                return clean_part.capitalize()
        
        print_substep("No name found, using default: 'there'")
        return "there"

if __name__ == '__main__':
    print_header("OpenServ Basic Agent Example")
    print_step("📚", "This example demonstrates core SDK concepts:", Fore.CYAN)
    print_substep("Agent: The main class for OpenServ platform communication")
    print_substep("Capability: Functions your agent can execute")
    print_substep("Schema: Type definitions using Pydantic")
    print_substep("Message Handling: Processing & responding to queries")
    
    # Set up custom logging configuration
    # Define a formatter that only shows the relevant information
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # For debug messages, add more context
            if record.levelno == logging.DEBUG:
                return f"DEBUG [{record.name}] {record.getMessage()}"
            # For info, show just brief info
            elif record.levelno == logging.INFO:
                msg = record.getMessage()
                # Abbreviate long messages
                if len(msg) > 100:
                    return f"INFO [{record.name}] {msg[:100]}..."
                return f"INFO [{record.name}] {msg}"
            # For warnings and errors, show full message
            else:
                return f"{record.levelname} [{record.name}] {record.getMessage()}"
    
    # Configure root logger with custom formatter
    root_handler = logging.StreamHandler()
    root_handler.setFormatter(CustomFormatter())
    
    # Reset root logger and add our handler
    root_logger = logging.getLogger()
    # Remove all existing handlers
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)
    root_logger.addHandler(root_handler)
    
    # Set default logging levels
    root_logger.setLevel(logging.WARNING)  # Only show warnings+ by default
    
    # Add our filter to prevent sensitive data exposure
    log_filter = SensitiveDataFilter()
    root_handler.addFilter(log_filter)
    
    # Enable INFO logging only for important loggers
    for logger_name in ['src.agent', 'src.capability', 'src.client']:
        logging.getLogger(logger_name).setLevel(logging.INFO)
    
    # Disable verbose libraries completely
    for logger_name in ['httpx', 'urllib3', 'asyncio']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Load system prompt from file but don't print its contents
    print_step("📄", "Loading system prompt from system_basic_agent.md", Fore.CYAN)
    with open("examples/system_basic_agent.md", "r") as f:
        system_prompt = f.read()
    print_substep("System prompt loaded successfully")
    
    print_header("Agent Initialization")
    # Create the agent with configuration
    print_step("⚙️", "Creating agent with configuration")
    agent = BasicAgent(
        AgentOptions(
            system_prompt=system_prompt,
            api_key=os.getenv('OPENSERV_API_KEY'),
            on_error=error_handler
        )
    )
    print_success("Agent instance created")
    
    # Add capabilities to the agent
    print_step("🧩", "Adding capabilities to agent")
    agent.add_capabilities([
        Capability(
            name="greet",
            description="Greet a user by name",
            schema=GreetArgs,
            run=lambda data, _: f"👋 Hello, {data.name}!\nHow can I help you today?"
        ),
        Capability(
            name="farewell",
            description="Say goodbye to a user",
            schema=FarewellArgs,
            run=lambda data, _: f"👋 Goodbye, {data.name}!\nHave a great day!"
        ),
        Capability(
            name="help",
            description="Show available commands",
            schema=HelpArgs,
            run=lambda _, __: "📋 Available Commands:\n\n• Hi / Hello - Greet the agent\n• Help - Show this help message\n• Bye / Goodbye - End the conversation"
        )
    ])
    
    print_success(f"Agent configured with {len(agent.tools)} capabilities")
    
    print_header("Starting Agent Server")
    print_step("💡", "Usage tips:", Fore.CYAN)
    print_substep("Try saying: 'hi', 'hello', 'help', or 'goodbye'")
    print_substep("You can add your name: 'Hi, I am Alex' or 'Hello, my name is Alex'")
    
    print_step("🚀", "Starting agent server...", Fore.GREEN)
    agent.start()
