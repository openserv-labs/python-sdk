"""
Marketing agent for OpenServ Python SDK.

Demonstrates a marketing agent with social media capabilities.
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

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verify API key
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError('OPENAI_API_KEY environment variable is required')

# Initialize OpenAI client
openai_client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    timeout=30.0  # Add a timeout to prevent hanging
)

# Define models for capabilities
class SocialMediaPlatform(str, Enum):
    TWITTER = 'twitter'
    LINKEDIN = 'linkedin'
    FACEBOOK = 'facebook'
    INSTAGRAM = 'instagram'

class SocialMediaPostParams(BaseModel):
    platform: SocialMediaPlatform
    topic: str

class TwitterPostParams(BaseModel):
    post_content: str

class EngagementMetrics(BaseModel):
    likes: int
    shares: int
    comments: int
    impressions: int

class AnalyzeEngagementParams(BaseModel):
    platform: SocialMediaPlatform
    metrics: EngagementMetrics

# Define capability functions
async def create_social_media_post(data, messages):
    """Creates a social media post for the specified platform."""
    args = data["args"]
    
    # Match the TypeScript implementation's prompting and model
    completion = openai_client.chat.completions.create(
        model='gpt-4o',  # Use the same model as in TypeScript version
        messages=[
            {
                'role': 'system',
                'content': f"""You are a marketing expert. Create a compelling {args.platform} post about: {args.topic}

Follow these platform-specific guidelines:
- Twitter: Max 280 characters, casual tone, use hashtags
- LinkedIn: Professional tone, industry insights, call to action
- Facebook: Engaging, conversational, can be longer
- Instagram: Visual focus, emoji-rich, hashtag clusters, brief

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

    generated_post = completion.choices[0].message.content
    logger.info(f"Generated {args.platform} post: {generated_post}")
    
    # Print directly to console for user visibility
    print(f"\n==== Social Media Post for {args.platform} about '{args.topic}' ====")
    print(generated_post)
    print("="*50)
    
    # If Twitter, notify the user we can post it with postToTwitter capability
    if args.platform.lower() == 'twitter':
        print("\nTo post this to Twitter, use the 'postToTwitter' capability.")
    
    return generated_post

async def post_to_twitter(data, messages):
    """Posts content to Twitter (mock implementation)."""
    args = data["args"]
    post_content = args.post_content
    
    # In a real implementation, this would use the Twitter API
    # For this example, we'll mock the API call
    logger.info(f"Would post to Twitter: {post_content}")
    
    # Print directly to console for user visibility
    print(f"\n==== Posting to Twitter ====")
    print(f"Content: {post_content}")
    print("Status: Success (simulated)")
    print("="*50)
    
    # Mock a successful response
    return json.dumps({
        "success": True,
        "post_id": "123456789",
        "platform": "twitter"
    })

async def analyze_engagement(data, messages):
    """Analyzes social media engagement metrics and provides recommendations."""
    args = data["args"]
    
    # Create a simple dictionary structure that matches the TypeScript implementation's expected format
    metrics_dict = {
        "platform": args.platform,
        "metrics": {
            "likes": args.metrics.likes,
            "shares": args.metrics.shares,
            "comments": args.metrics.comments,
            "impressions": args.metrics.impressions
        }
    }
    
    # Match the TypeScript implementation's prompting and model
    completion = openai_client.chat.completions.create(
        model='gpt-4o',  # Use the same model as in TypeScript version
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
                'content': json.dumps(metrics_dict)
            }
        ]
    )

    analysis = completion.choices[0].message.content
    logger.info(f"Generated engagement analysis for {args.platform}: {analysis}")
    
    # Print directly to console for user visibility
    print(f"\n==== Engagement Analysis for {args.platform} ====")
    print(analysis)
    print("="*50)
    
    return analysis

# Verify system prompt file
system_prompt_path = Path(__file__).parent.joinpath('system.md')
if not system_prompt_path.exists():
    raise FileNotFoundError(f"System prompt file not found at {system_prompt_path}")

# Load system prompt
system_prompt = system_prompt_path.read_text()
logger.info(f"Loaded system prompt ({len(system_prompt)} characters)")
logger.info(f"System prompt first 100 chars: {system_prompt[:100]}...")

# Create agent
marketing_manager = Agent(
    AgentOptions(
        system_prompt=system_prompt,
        api_key=os.getenv('OPENSERV_API_KEY'),
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model="gpt-4o",  # Match the TypeScript version
        on_error=lambda e, context: logger.error(f"Agent error: {str(e)}\nContext: {json.dumps(context, default=str)}")
    )
)

# Add capabilities to the agent
marketing_manager.add_capabilities([
    Capability(
        name='createSocialMediaPost',
        description='Creates a social media post for the specified platform',
        schema=SocialMediaPostParams,
        run=create_social_media_post
    ),
    Capability(
        name='postToTwitter',
        description='Posts content to Twitter',
        schema=TwitterPostParams,
        run=post_to_twitter
    ),
    Capability(
        name='analyzeEngagement',
        description='Analyzes social media engagement metrics and provides recommendations',
        schema=AnalyzeEngagementParams,
        run=analyze_engagement
    )
])

if __name__ == '__main__':
    # Set lower log level for HTTP libraries
    for logger_name in ['httpx', 'urllib3']:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    try:
        print("\n🚀 Starting marketing agent...")
        print("💡 Note: For task completion, the agent will use its built-in capabilities.")
        
        # Log capabilities for debugging
        logger.info(f"Agent has {len(marketing_manager.tools)} capabilities: {[t.name for t in marketing_manager.tools]}")
        
        marketing_manager.start()
    except Exception as e:
        logger.error(f"Error starting agent: {e}")
        import sys
        sys.exit(1)
