"""
Marketing agent implementation for the OpenServ Python SDK.
This example demonstrates how to create a marketing-focused agent with social media capabilities.

Key concepts demonstrated:
1. Creating a custom agent by extending the base Agent class
2. Defining and using Pydantic models for structured data
3. Using the process() method for local OpenAI testing
4. Working with OpenServ's Twitter integration
5. Handling chat messages and sending responses
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
import openai
import logging
from src.types import RespondChatMessageAction, ProcessParams

from src import Agent, Capability
from src import AgentOptions

# Load environment variables from .env file
# This loads API keys for OpenServ and OpenAI
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# -------------------------------------------------
# DEFINE DATA MODELS USING PYDANTIC
# -------------------------------------------------
# Using Pydantic models provides:
# - Type validation
# - Automatic data parsing
# - Clear structure for parameters
# -------------------------------------------------

class SocialMediaPostParams(BaseModel):
    """
    Parameters for creating a social media post.
    This model defines the structure expected by the createSocialMediaPost capability.
    """
    platform: str  # The social media platform (e.g., Twitter, LinkedIn)
    topic: str     # The topic to post about

class EngagementMetrics(BaseModel):
    """
    Model for social media engagement metrics.
    These metrics are used for analyzing post performance.
    """
    likes: int
    shares: int
    comments: int
    impressions: int

class AnalyzeEngagementParams(BaseModel):
    """
    Parameters for analyzing engagement metrics.
    This combines the platform with the metrics data.
    """
    platform: str
    metrics: EngagementMetrics

# -------------------------------------------------
# LOAD SYSTEM PROMPT
# -------------------------------------------------
# The system prompt defines the agent's personality and instructions.
# It's stored in a separate file for better organization.
# -------------------------------------------------

system_prompt_path = Path(__file__).parent.joinpath('system.md')
if not system_prompt_path.exists():
    raise FileNotFoundError("system.md not found")
system_prompt = system_prompt_path.read_text()

# -------------------------------------------------
# CUSTOM AGENT IMPLEMENTATION
# -------------------------------------------------
# By extending the Agent class, we can customize:
# - How chat messages are processed
# - How the agent responds to specific triggers
# - Integration with external APIs (like Twitter)
# -------------------------------------------------

class MarketingAgent(Agent):
    async def respond_to_chat(self, action: RespondChatMessageAction) -> None:
        """
        Handle chat messages for the marketing agent.
        
        This method is called whenever the agent receives a message from a user.
        The flow is:
        1. User sends message to OpenServ platform
        2. OpenServ forwards the message to your agent via this method
        3. Agent processes the message and generates a response
        4. Agent sends the response back to OpenServ
        5. OpenServ delivers the response to the user
        
        The action parameter contains:
        - messages: The conversation history
        - me: Information about the agent itself
        - workspace: Information about the workspace
        - integrations: Available integrations (including Twitter)
        """
        try:
            logger.info(f"Received chat action: {action}")
            
            if not action.messages:
                logger.warning("No messages in action")
                return

            # Extract the most recent message from the user
            last_message = action.messages[-1].message
            logger.info(f"Processing message: {last_message}")
            response = None
            
            # Check if we're running in local mode
            is_local_mode = os.getenv('OPENAI_API_KEY') and (not os.getenv('OPENSERV_API_KEY') or not action.integrations)
            
            if is_local_mode:
                logger.info("Running in local mode with OpenAI API")

            if 'create post' in last_message.lower() or 'social media' in last_message.lower():
                logger.info("Detected social media post request")
                
                platform = 'twitter'  # Default platform
                topic = last_message.split('about')[-1].strip() if 'about' in last_message else last_message
                logger.info(f"Extracted platform: {platform}, topic: {topic}")
                
                # Local testing with OpenAI API
                if is_local_mode:
                    logger.info("Generating social media post with local OpenAI API")
                    result = await self.process(ProcessParams(
                        messages=[
                            {
                                'role': 'system',
                                'content': f"You are a marketing expert. Create a compelling social media post for {platform} about {topic}."
                            },
                            {
                                'role': 'user',
                                'content': f"Create a {platform} post about: {topic}"
                            }
                        ]
                    ))
                    
                    # Extract the response content
                    if result and 'response' in result:
                        response = result['response']
                else:
                    # Here we would use the Twitter integration through OpenServ
                    # But for now, still use the local approach
                    result = await self.process(ProcessParams(
                        messages=[
                            {
                                'role': 'system',
                                'content': f"You are a marketing expert. Create a compelling social media post for {platform} about {topic}."
                            },
                            {
                                'role': 'user',
                                'content': f"Create a {platform} post about: {topic}"
                            }
                        ]
                    ))
                    
                    if result and 'response' in result:
                        response = result['response']
            
            elif 'analyze' in last_message.lower() or 'engagement' in last_message.lower():
                logger.info("Detected engagement analysis request")
                
                # Extract metrics from the message if provided
                likes = None
                shares = None
                comments = None
                impressions = None
                
                # Try to parse metrics from the message
                import re
                likes_match = re.search(r'(\d+)\s*likes', last_message, re.IGNORECASE)
                shares_match = re.search(r'(\d+)\s*shares', last_message, re.IGNORECASE)
                comments_match = re.search(r'(\d+)\s*comments', last_message, re.IGNORECASE)
                impressions_match = re.search(r'(\d+)\s*impressions', last_message, re.IGNORECASE)
                
                # Get values if they exist in the message
                if likes_match:
                    likes = int(likes_match.group(1))
                if shares_match:
                    shares = int(shares_match.group(1))
                if comments_match:
                    comments = int(comments_match.group(1))
                if impressions_match:
                    impressions = int(impressions_match.group(1))
                
                # Use random values for any missing metrics
                import random
                likes = likes if likes is not None else random.randint(10, 1000)
                shares = shares if shares is not None else random.randint(5, 500)
                comments = comments if comments is not None else random.randint(1, 200)
                impressions = impressions if impressions is not None else random.randint(100, 10000)
                
                # Create metrics object with extracted or random values
                metrics = EngagementMetrics(
                    likes=likes, 
                    shares=shares, 
                    comments=comments, 
                    impressions=impressions
                )
                logger.info(f"Using metrics: {metrics}")
                
                # Process locally with OpenAI API
                result = await self.process(ProcessParams(
                    messages=[
                        {
                            'role': 'system',
                            'content': "You are a social media analytics expert. Analyze the engagement metrics and provide actionable recommendations."
                        },
                        {
                            'role': 'user',
                            'content': f"Platform: twitter\nMetrics: {metrics.model_dump_json()}"
                        }
                    ]
                ))
                
                # Extract response
                if result and 'response' in result:
                    response = result['response']

            # Default response if no command is detected
            if not response:
                logger.info("No specific command detected, sending default response")
                response = "I'm a marketing agent that can create social media posts and analyze engagement metrics. Try asking me to create a post or analyze engagement!"

            logger.info(f"Sending response: {response}")
            
            # Handle sending the message differently depending on mode
            if is_local_mode:
                # In local mode, just log the response instead of trying to send it
                logger.info(f"LOCAL MODE RESPONSE: {response}")
                print(f"\nAgent response: {response}\n")
            else:
                # In platform mode, use the send_message method
                try:
                    await self.send_message(response)
                except Exception as send_error:
                    logger.error(f"Error sending message: {str(send_error)}")
                    # Try a backup approach if needed
                    if hasattr(action, 'workspace') and hasattr(action, 'me'):
                        try:
                            await self.send_chat_message(
                                workspace_id=action.workspace.id,
                                agent_id=action.me.id,
                                message=response
                            )
                        except Exception as backup_error:
                            logger.error(f"Backup sending failed: {str(backup_error)}")

        except Exception as e:
            logger.error(f"Error in respond_to_chat: {str(e)}", exc_info=True)
            # Send error message to user
            try:
                # Check if we're in local mode
                is_local_mode = os.getenv('OPENAI_API_KEY') and (not os.getenv('OPENSERV_API_KEY') or not action.integrations)
                
                if is_local_mode:
                    # Just log the error in local mode
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    logger.error(f"LOCAL MODE ERROR: {error_msg}")
                    print(f"\nAgent error: {error_msg}\n")
                else:
                    # Try to send the error message in platform mode
                    await self.send_message(f"Sorry, I encountered an error: {str(e)}")
            except Exception as send_error:
                logger.error(f"Failed to send error message: {str(send_error)}")

# -------------------------------------------------
# CREATE_AGENT FUNCTION EXPLAINED
# -------------------------------------------------
# The create_agent() function below:
# 
# 1. Creates a new MarketingAgent instance with:
#    - The system prompt loaded from system.md
#    - The OpenServ API key for platform communication
#    - The OpenAI API key for local testing
#    - The specified model (gpt-4o)
#
# 2. Registers two capabilities:
#    - createSocialMediaPost: For generating platform-specific posts
#    - analyzeEngagement: For analyzing social media metrics
#
# This factory pattern makes testing easier by separating
# object creation from business logic.
# -------------------------------------------------

def create_agent() -> MarketingAgent:
    """
    Create and configure the marketing agent.
    
    This factory function:
    1. Creates a new MarketingAgent instance
    2. Configures it with necessary options
    3. Adds capabilities to the agent
    4. Returns the fully configured agent
    
    The API keys are loaded from environment variables:
    - OPENSERV_API_KEY: For authenticating with the OpenServ platform
    - OPENAI_API_KEY: For local testing with OpenAI
    """
    
    # Create agent with options
    agent = MarketingAgent(
        AgentOptions(
            system_prompt=system_prompt,              # Personality/instructions
            api_key=os.getenv('OPENSERV_API_KEY'),    # For OpenServ platform API
            openai_api_key=os.getenv('OPENAI_API_KEY'), # For local OpenAI testing
            model="gpt-4o"                            # The OpenAI model to use
        )
    )

    # -------------------------------------------------
    # CAPABILITY REGISTRATION
    # -------------------------------------------------
    # Capabilities define the specific functions your agent can perform.
    # Each capability has:
    # - A name: Used by the LLM to identify the function
    # - A description: Helps the LLM understand when to use it
    # - A schema: Defines the expected input parameters
    # - A run function: The actual implementation
    # -------------------------------------------------
    
    agent.add_capabilities([
        # Social media post creation capability
        Capability(
            name='createSocialMediaPost',
            description='Creates a social media post for the specified platform',
            schema=SocialMediaPostParams,
            run=create_social_media_post
        ),
        
        # Engagement analysis capability
        Capability(
            name='analyzeEngagement',
            description='Analyzes social media engagement metrics and provides recommendations',
            schema=AnalyzeEngagementParams,
            run=analyze_engagement
        )
    ])

    return agent

# -------------------------------------------------
# CAPABILITY IMPLEMENTATIONS
# -------------------------------------------------
# These functions implement the actual functionality of each capability.
# They're defined separately from the agent class for better organization.
# -------------------------------------------------

async def create_social_media_post(params, _):
    """
    Creates a social media post for the specified platform.
    
    This function:
    1. Validates the input parameters using Pydantic
    2. Generates content for the specified platform
    3. Returns the post content as a string
    
    In a production environment with OpenServ:
    - This would format the post for the platform
    - The agent would post it via the platform integration
    
    For local testing with OpenAI:
    - This directly generates the content using the OpenAI API
    """
    try:
        # Extract and validate parameters using Pydantic
        # The params object might have the parameters directly or in an 'args' field
        if isinstance(params, dict) and 'args' in params:
            # If params has an 'args' field (common when called from the OpenAI function calling system)
            # In this case, 'args' might be a JSON string or a dict
            args_data = params['args']
            if isinstance(args_data, str):
                import json
                args_data = json.loads(args_data)
            args = SocialMediaPostParams.model_validate(args_data)
        else:
            # If params is already the expected structure
            args = SocialMediaPostParams.model_validate(params)
            
        platform = args.platform
        topic = args.topic
        
        logger.info(f"Creating social media post for platform: {platform}, topic: {topic}")

        # For local testing with OpenAI
        if os.getenv('OPENAI_API_KEY'):
            openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a marketing expert. Create a compelling {platform} post about: {topic}

Follow these platform-specific guidelines:
- Twitter: Max 280 characters, casual tone, use hashtags
- LinkedIn: Professional tone, industry insights, call to action
- Facebook: Engaging, conversational, can be longer

Include emojis where appropriate. Focus on driving engagement.

Only generate post for the given platform. Don't generate posts for other platforms."""
                    },
                    {
                        "role": "user",
                        "content": topic
                    }
                ]
            )
            return completion.choices[0].message.content
        
        return f"Created a {platform} post about: {topic}"

    except Exception as e:
        logger.error(f"Error creating social media post: {str(e)}", exc_info=True)
        return f"Error creating social media post: {str(e)}"

async def analyze_engagement(params, _):
    """
    Analyze social media engagement metrics and provide recommendations.
    
    This function:
    1. Validates the input parameters using Pydantic
    2. Calculates engagement rates based on the metrics
    3. Provides analysis and recommendations
    
    In a production environment with OpenServ:
    - Metrics would come from the platform's Twitter integration
    - Analysis would be based on real-time data
    
    For local testing with OpenAI:
    - This directly generates the analysis using the OpenAI API
    """
    try:
        # Extract and validate parameters, handling different parameter formats
        if isinstance(params, dict) and 'args' in params:
            # If params has an 'args' field
            args_data = params['args']
            if isinstance(args_data, str):
                import json
                args_data = json.loads(args_data)
            args = AnalyzeEngagementParams.model_validate(args_data)
        else:
            # If params is already the expected structure
            args = AnalyzeEngagementParams.model_validate(params)
            
        logger.info(f"Processing engagement analysis for platform: {args.platform}")
        
        # For local testing with OpenAI
        if os.getenv('OPENAI_API_KEY'):
            openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            completion = openai_client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {
                        'role': 'system',
                        'content': """You are a social media analytics expert. Analyze the engagement metrics and provide actionable recommendations.

Consider platform-specific benchmarks:
- Twitter: Engagement rate = (likes + shares + comments) / impressions
- LinkedIn: Engagement rate = (likes + shares + comments) / impressions * 100
- Facebook: Engagement rate = (likes + shares + comments) / impressions * 100

Provide:
1. Current engagement rate
2. Performance assessment (below average, average, above average)
3. Top 3 actionable recommendations to improve engagement
4. Key metrics to focus on for improvement"""
                    },
                    {
                        'role': 'user',
                        'content': f"Platform: {args.platform}\nMetrics: {args.metrics.model_dump_json()}"
                    }
                ]
            )
            return completion.choices[0].message.content
        
        return f"Analyzed engagement metrics for {args.platform}"

    except Exception as e:
        logger.error(f"Error analyzing engagement: {str(e)}", exc_info=True)
        return f"Error analyzing engagement: {str(e)}"

# -------------------------------------------------
# SCRIPT ENTRY POINT
# -------------------------------------------------
# This section runs when the script is executed directly.
# It doesn't run when the file is imported as a module.
# -------------------------------------------------

if __name__ == '__main__':
    # Configure logging to show more details
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting marketing agent...")
    
    # Create and start the agent
    agent = create_agent()
    agent.start()  # This starts the HTTP server to listen for incoming requests from OpenServ
