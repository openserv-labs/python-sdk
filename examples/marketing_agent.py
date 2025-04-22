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
from src.types import RespondChatMessageAction

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

            # -------------------------------------------------
            # MESSAGE PROCESSING LOGIC
            # -------------------------------------------------
            # This section demonstrates two approaches:
            # 1. Local processing with OpenAI API (great for development)
            # 2. Using integrations through OpenServ (for production)
            # -------------------------------------------------

            if 'create post' in last_message.lower() or 'social media' in last_message.lower():
                logger.info("Detected social media post request")
                
                # Simple NLP to extract details from the message
                platform = 'twitter'  # Default platform
                topic = last_message.split('about')[-1].strip() if 'about' in last_message else last_message
                logger.info(f"Extracted platform: {platform}, topic: {topic}")
                
                # -------------------------------------------------
                # LOCAL TESTING APPROACH
                # -------------------------------------------------
                # This uses process() to run with the OpenAI API directly
                # Great for development and testing without platform dependency
                # -------------------------------------------------
                
                result = await self.process({
                    'messages': [
                        {
                            'role': 'system',
                            'content': f"You are a marketing expert. Create a compelling social media post for {platform} about {topic}."
                        },
                        {
                            'role': 'user',
                            'content': f"Create a {platform} post about: {topic}"
                        }
                    ]
                })
                
                # Extract the response content
                if result and 'response' in result:
                    response = result['response']
                    
                # -------------------------------------------------
                # TWITTER INTEGRATION APPROACH
                # -------------------------------------------------
                # In a production environment, you would use the Twitter
                # integration provided by OpenServ to post directly to Twitter.
                # 
                # Example (uncomment to use):
                # if any(i.provider == "twitter" for i in action.integrations):
                #     # Create post content using the response
                #     tweet_content = response
                #     # Call the Twitter integration to post the tweet
                #     tweet_result = await self.call_integration({
                #         "workspace_id": action.workspace.id,
                #         "integration_id": "twitter-v2",
                #         "details": {
                #             "endpoint": "/2/tweets",
                #             "method": "POST",
                #             "data": {
                #                 "text": tweet_content
                #             }
                #         }
                #     })
                #     # Add the tweet info to the response
                #     response += f"\n\nPost has been published to Twitter!"
                # -------------------------------------------------
            
            elif 'analyze' in last_message.lower() or 'engagement' in last_message.lower():
                logger.info("Detected engagement analysis request")
                
                # Example metrics - in real usage, these would be extracted from the message
                # or retrieved from the Twitter API using the integration
                metrics = EngagementMetrics(likes=100, shares=50, comments=30, impressions=1000)
                logger.info(f"Using metrics: {metrics}")
                
                # Process locally with OpenAI API
                result = await self.process({
                    'messages': [
                        {
                            'role': 'system',
                            'content': "You are a social media analytics expert. Analyze the engagement metrics and provide actionable recommendations."
                        },
                        {
                            'role': 'user',
                            'content': f"Platform: twitter\nMetrics: {metrics.model_dump_json()}"
                        }
                    ]
                })
                
                # Extract response
                if result and 'response' in result:
                    response = result['response']
                    
                # -------------------------------------------------
                # TWITTER METRICS INTEGRATION
                # -------------------------------------------------
                # For real production use, you would retrieve actual metrics:
                # 
                # Example (uncomment to use):
                # if any(i.provider == "twitter" for i in action.integrations):
                #     # Get tweet metrics from Twitter API
                #     twitter_metrics = await self.call_integration({
                #         "workspace_id": action.workspace.id,
                #         "integration_id": "twitter-v2",
                #         "details": {
                #             "endpoint": "/2/tweets/{id}/metrics",
                #             "method": "GET",
                #             "params": {"id": "tweet_id_here"}
                #         }
                #     })
                #     # Process the real metrics...
                # -------------------------------------------------

            # Default response if no command is detected
            if not response:
                logger.info("No specific command detected, sending default response")
                response = "I'm a marketing agent that can create social media posts and analyze engagement metrics. Try asking me to create a post or analyze engagement!"

            logger.info(f"Sending response: {response}")
            # Send response back to the user using the SDK's method
            # This uses the send_message convenience method that extracts
            # workspace_id and agent_id from the current context
            await self.send_message(response)

        except Exception as e:
            logger.error(f"Error in respond_to_chat: {str(e)}", exc_info=True)
            # Send error message to user
            await self.send_message(f"Sorry, I encountered an error: {str(e)}")

# -------------------------------------------------
# AGENT FACTORY FUNCTION
# -------------------------------------------------
# This pattern separates agent creation from execution,
# making it easier to test and configure.
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
        
        # In production, this would be handled by the OpenServ platform
        # and the result would be posted via the Twitter integration
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
        # Extract and validate parameters
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
        
        # In production, this would be handled by the OpenServ platform
        # with real Twitter metrics data
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
