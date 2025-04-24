from typing import TypeVar, Protocol, Dict, Any, List, Awaitable, Union, Generic
from pydantic import BaseModel
import inspect
import json
import logging
from .types import AgentAction, ChatMessage

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

class CapabilityFunction(Protocol[T]):
    """Protocol defining the expected signature of a capability's run function."""
    def __call__(
        self,
        params: Dict[str, Union[T, AgentAction]],
        messages: List[Dict[str, Any]]
    ) -> Union[str, Awaitable[str]]: ...

class Capability(Generic[T]):
    """
    A capability that can be added to an agent.
    
    Attributes:
        name: The unique name of the capability
        description: A description of what the capability does
        schema: The Pydantic model class defining the capability's parameters
        run: The function that implements the capability's behavior
    """
    def __init__(
        self,
        name: str,
        description: str,
        schema: type[T],
        run: CapabilityFunction[T]
    ) -> None:
        """
        Initialize a new Capability instance.
        
        Args:
            name: The name of the capability
            description: A description of what the capability does
            schema: The Pydantic model class defining the capability's parameters
            run: The function that implements the capability's behavior
            
        Raises:
            TypeError: If schema is not a Pydantic model class
            ValueError: If run is not callable
        """
        if not issubclass(schema, BaseModel):
            raise TypeError("schema must be a Pydantic model class")
        if not callable(run):
            raise ValueError("run must be a callable")
            
        self.name = name
        self.description = description
        self.schema = schema
        
        # Ensure run is an async function
        if inspect.iscoroutinefunction(run):
            self._run = run
        else:
            # Convert sync function to async
            async def async_run(args, messages) -> str:
                return run(args, messages)
            self._run = async_run
            
    async def run(self, params: Dict[str, Any] | BaseModel, messages: List[Any]) -> str:
        """
        Execute the capability with the given parameters.
        
        This method handles parsing JSON arguments from OpenAI and passing them to the capability's run function.
        
        Args:
            params: A dictionary with the arguments for the capability, or a Pydantic model instance
            messages: The conversation history, can be Dict[str, Any] or custom message types
            
        Returns:
            The result of executing the capability
        """
        try:
            # Handle both dictionary and Pydantic model inputs for params
            if isinstance(params, dict):
                args = params.get('args', {})
                action = params.get('action')
                
                # If args is a string (JSON), parse it
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON arguments: {e}")
                        return f"Error: Invalid JSON arguments: {str(e)}"
                        
                # If args is already a Pydantic model instance, use it directly
                if not isinstance(args, BaseModel):
                    try:
                        # Validate args with the schema
                        validated_args = self.schema(**args)
                    except Exception as e:
                        logger.error(f"Validation error for {self.name}: {str(e)}")
                        return f"Error: Invalid arguments: {str(e)}"
                else:
                    validated_args = args
            elif isinstance(params, BaseModel):
                # If params is already a BaseModel (Pydantic), use it directly
                validated_args = params
                action = None
            else:
                raise TypeError(f"Expected dict or BaseModel, got {type(params)}")
            
            # Ensure messages are properly formatted
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    formatted_messages.append(msg)
                elif hasattr(msg, 'dict'):
                    # If message is a Pydantic model, convert to dict
                    formatted_messages.append(msg.dict())
                else:
                    # Otherwise just keep as is
                    formatted_messages.append(msg)
            
            # Execute the capability's run function
            result = await self._run({"args": validated_args, "action": action}, formatted_messages)
            return result
        except Exception as e:
            logger.exception(f"Error executing capability {self.name}: {str(e)}")
            return f"Error: {str(e)}"
