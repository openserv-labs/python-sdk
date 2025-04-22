"""
Marketing agent implementation for the OpenServ Python SDK.
This example demonstrates how to create a marketing-focused agent with social media capabilities.
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

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Define schemas using Pydantic
class SocialMediaPostParams(BaseModel):
    platform: str
    topic: str

class EngagementMetrics(BaseModel):
    likes: int
    shares: int
    comments: int
    impressions: int

class AnalyzeEngagementParams(BaseModel):
    platform: str
    metrics: EngagementMetrics

# Load system prompt
system_prompt_path = Path(__file__).parent.joinpath('system.md')
if not system_prompt_path.exists():
    raise FileNotFoundError("system.md not found")
system_prompt = system_prompt_path.read_text()

def create_agent() -> MarketingAgent:
    """Create and configure the marketing agent"""
    
    # Create agent with options
    agent = MarketingAgent(
        AgentOptions(
            system_prompt=system_prompt,
            api_key=os.getenv('OPENSERV_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            model="gpt-4o"
        )
    )

    # Add capabilities
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

async def create_social_media_post(params, _):
    """Creates a social media post for the specified platform."""
    try:
        # Extract and validate parameters
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
        return f"Created a {platform} post about: {topic}"

    except Exception as e:
        logger.error(f"Error creating social media post: {str(e)}", exc_info=True)
        return f"Error creating social media post: {str(e)}"

async def analyze_engagement(params, _):
    """Analyze social media engagement metrics."""
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
        return f"Analyzed engagement metrics for {args.platform}"

    except Exception as e:
        logger.error(f"Error analyzing engagement: {str(e)}", exc_info=True)
        return f"Error analyzing engagement: {str(e)}"

# Custom agent class with message handling
class MarketingAgent(Agent):
    async def respond_to_chat(self, action: RespondChatMessageAction) -> None:
        """Handle chat messages for the marketing agent"""
        try:
            logger.info(f"Received chat action: {action}")
            
            if not action.messages:
                logger.warning("No messages in action")
                return

            last_message = action.messages[-1].message
            logger.info(f"Processing message: {last_message}")
            response = None

            # Process the message using the SDK's capabilities
            if 'create post' in last_message.lower() or 'social media' in last_message.lower():
                logger.info("Detected social media post request")
                # Extract platform and topic from message
                platform = 'twitter'  # Default platform
                topic = last_message.split('about')[-1].strip() if 'about' in last_message else last_message
                logger.info(f"Extracted platform: {platform}, topic: {topic}")
                response = await self.process(SocialMediaPostParams(platform=platform, topic=topic))
            
            elif 'analyze' in last_message.lower() or 'engagement' in last_message.lower():
                logger.info("Detected engagement analysis request")
                # Example metrics - in real usage, these would come from the message
                metrics = EngagementMetrics(likes=100, shares=50, comments=30, impressions=1000)
                logger.info(f"Using metrics: {metrics}")
                response = await self.process(AnalyzeEngagementParams(platform='twitter', metrics=metrics))

            # Default response if no command is detected
            if not response:
                logger.info("No specific command detected, sending default response")
                response = "I'm a marketing agent that can create social media posts and analyze engagement metrics. Try asking me to create a post or analyze engagement!"

            logger.info(f"Sending response: {response}")
            # Send response back to the user using the SDK's method
            await self.send_message(response)

        except Exception as e:
            logger.error(f"Error in respond_to_chat: {str(e)}", exc_info=True)
            # Send error message to user
            await self.send_message(f"Sorry, I encountered an error: {str(e)}")

if __name__ == '__main__':
    # Configure logging to show more details
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting marketing agent...")
    
    # Create and start the agent
    agent = create_agent()
    agent.start()
