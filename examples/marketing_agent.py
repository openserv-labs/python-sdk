"""
📚 Marketing Agent Example for OpenServ Python SDK 📚

Demonstrates a marketing agent with social media capabilities, including Twitter native integration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
import openai
import logging
import json
from enum import Enum
from typing import Dict, Any, List, Optional

from src import Agent, Capability, AgentOptions

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define models for capabilities
class SocialMediaPlatform(str, Enum):
    """Supported social media platforms."""
    TWITTER = 'twitter'
    LINKEDIN = 'linkedin'
    FACEBOOK = 'facebook'

class SocialMediaPostParams(BaseModel):
    """Parameters for creating a social media post."""
    platform: SocialMediaPlatform
    topic: str

class GetTwitterAccountParams(BaseModel):
    """Parameters for getting Twitter account info (empty schema)."""
    pass

class SendMarketingTweetParams(BaseModel):
    """Parameters for sending a marketing tweet."""
    tweetText: str

class EngagementMetrics(BaseModel):
    """Social media engagement metrics structure."""
    likes: int
    shares: int
    comments: int
    impressions: int

class AnalyzeEngagementParams(BaseModel):
    """Parameters for analyzing social media engagement."""
    platform: SocialMediaPlatform
    metrics: EngagementMetrics

# Define capability functions
async def create_social_media_post(data, messages):
    """
    Creates a social media post for the specified platform.
    """
    args = data["args"]
    platform = str(args.platform)
    
    # Create completion using OpenAI
    completion = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')).chat.completions.create(
        model='gpt-4o',
        messages=[
            {
                'role': 'system',
                'content': f"""You are a marketing expert. Create a compelling {platform} post about: {args.topic}

Follow these platform-specific guidelines:
- Twitter: Max 280 characters, casual tone, use hashtags
- LinkedIn: Professional tone, industry insights, call to action
- Facebook: Engaging, conversational, can be longer

Include emojis where appropriate. Focus on driving engagement.

Only generate post for the given platform. Don't generate posts for other platforms.
"""
            },
            {
                'role': 'user',
                'content': args.topic
            }
        ]
    )

    return completion.choices[0].message.content

async def get_twitter_account(data, messages):
    """
    Gets the Twitter account for the current user.
    
    Uses the Twitter API integration to retrieve account information.
    Requires proper workspace and integration setup.
    """
    # The agent instance is needed to call integrations
    agent = data.get("_agent")
    if not agent:
        return "Error: Twitter integration not available."
    
    # Get workspace from action context
    action = data.get("action")
    if not action or not hasattr(action, "workspace") or not hasattr(action.workspace, "id"):
        return "Error: Workspace information not available."
    
    # Call Twitter API using integration
    try:
        result = await agent.call_integration({
            'workspace_id': action.workspace.id,
            'integration_id': 'twitter-v2',
            'details': {
                'endpoint': '/2/users/me',
                'method': 'GET'
            }
        })
        
        # Extract username from result
        if hasattr(result, "output") and hasattr(result.output, "data"):
            return result.output.data.username
        
        return "Error: Unexpected response format from Twitter API."
    except Exception as e:
        logger.error(f"Twitter API error: {str(e)}")
        return f"Error retrieving Twitter account: {str(e)}"

async def send_marketing_tweet(data, messages):
    """
    Sends a marketing tweet to Twitter.
    
    Uses the Twitter API integration to post a tweet.
    Requires proper workspace and integration setup.
    """
    args = data["args"]
    
    # The agent instance is needed to call integrations
    agent = data.get("_agent")
    if not agent:
        return "Error: Twitter integration not available."
    
    # Get workspace from action context
    action = data.get("action")
    if not action or not hasattr(action, "workspace") or not hasattr(action.workspace, "id"):
        return "Error: Workspace information not available."
    
    # Call Twitter API to post the tweet
    try:
        result = await agent.call_integration({
            'workspace_id': action.workspace.id,
            'integration_id': 'twitter-v2',
            'details': {
                'endpoint': '/2/tweets',
                'method': 'POST',
                'data': {
                    'text': args.tweetText
                }
            }
        })
        
        # Extract tweet text from result
        if hasattr(result, "output") and hasattr(result.output, "data"):
            return result.output.data.text
        
        return "Error: Unexpected response format from Twitter API."
    except Exception as e:
        logger.error(f"Twitter API error: {str(e)}")
        return f"Error sending tweet: {str(e)}"

async def analyze_engagement(data, messages):
    """
    Analyzes social media engagement metrics and provides recommendations.
    
    Uses OpenAI to analyze engagement metrics and provide actionable insights.
    """
    args = data["args"]
    
    # Prepare data for analysis
    metrics_data = {
        'platform': str(args.platform),
        'metrics': {
            'likes': args.metrics.likes,
            'shares': args.metrics.shares,
            'comments': args.metrics.comments,
            'impressions': args.metrics.impressions
        }
    }
    
    # Create completion using OpenAI
    completion = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')).chat.completions.create(
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
                'content': json.dumps(metrics_data)
            }
        ]
    )

    return completion.choices[0].message.content

def create_agent() -> Agent:
    """Create and configure the marketing agent."""
    # Load system prompt
    system_prompt_path = Path(__file__).parent.joinpath('system.md')
    if not system_prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found at {system_prompt_path}")
    
    # Verify API key
    api_key = os.getenv('OPENSERV_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError('OPENSERV_API_KEY environment variable is required')
    if not openai_api_key:
        raise ValueError('OPENAI_API_KEY environment variable is required')
    
    # Create agent with configuration
    agent = Agent(
        AgentOptions(
            system_prompt=system_prompt_path.read_text(),
            api_key=api_key,
            openai_api_key=openai_api_key,
            model="gpt-4o"
        )
    )
    
    # Add all capabilities
    agent.add_capabilities([
        Capability(
            name='createSocialMediaPost',
            description='Creates a social media post for the specified platform',
            schema=SocialMediaPostParams,
            run=create_social_media_post
        ),
        Capability(
            name='getTwitterAccount',
            description='Gets the Twitter account for the current user',
            schema=GetTwitterAccountParams,
            run=get_twitter_account
        ),
        Capability(
            name='sendMarketingTweet',
            description='Sends a marketing tweet to Twitter',
            schema=SendMarketingTweetParams,
            run=send_marketing_tweet
        ),
        Capability(
            name='analyzeEngagement',
            description='Analyzes social media engagement metrics and provides recommendations',
            schema=AnalyzeEngagementParams,
            run=analyze_engagement
        )
    ])
    
    return agent

# Add a utility to normalize platform inputs for case-insensitivity
def add_case_insensitive_handling(agent):
    """
    Adds case-insensitive platform handling to the agent.
    
    This ensures that platform names like "Twitter", "TWITTER", and "twitter"
    all work correctly by normalizing them to lowercase.
    """
    original_handle_tool_route = agent.handle_tool_route
    
    async def normalized_handle_tool_route(tool_name, body):
        try:
            # Handle platform case-sensitivity for social media tools
            if isinstance(body, dict) and 'args' in body and isinstance(body['args'], dict):
                if tool_name == 'createSocialMediaPost' and 'platform' in body['args']:
                    if isinstance(body['args']['platform'], str):
                        body['args']['platform'] = body['args']['platform'].lower()
                
                elif tool_name == 'analyzeEngagement' and 'platform' in body['args']:
                    if isinstance(body['args']['platform'], str):
                        body['args']['platform'] = body['args']['platform'].lower()
            
            # Process normally with normalized values
            return await original_handle_tool_route(tool_name, body)
        except Exception as e:
            logger.error(f"Error in {tool_name}: {str(e)}")
            return {'result': f"Error processing {tool_name}: {str(e)}"}
    
    # Replace the original method
    agent.handle_tool_route = normalized_handle_tool_route
    return agent

if __name__ == '__main__':
    # Set lower log level for HTTP libraries
    for logger_name in ['httpx', 'urllib3']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    try:
        # Create and configure the agent
        marketing_manager = create_agent()
        
        # Add case-insensitive platform handling
        marketing_manager = add_case_insensitive_handling(marketing_manager)
        
        logger.info("Starting marketing agent with Twitter capabilities")
        logger.info("Note: Twitter capabilities require integration with OpenServ platform")
        
        # Start the agent
        marketing_manager.start()
    except Exception as e:
        logger.error(f"Error starting agent: {e}")
        import sys
        sys.exit(1)
