#!/usr/bin/env python3
"""
Agent Zero Integration for Intelligent Rate Limiter
Seamless integration that replaces Agent Zero's static rate limiting with intelligent self-tuning
"""

import asyncio
import time
import logging
from typing import Optional, Callable, Awaitable
from datetime import datetime, timezone
import models
from python.helpers.tokens import approximate_tokens
from intelligent_rate_limiter import (
    get_intelligent_rate_limiter_for_model,
    record_api_usage,
    intelligent_rate_limiter
)

logger = logging.getLogger(__name__)


class IntelligentRateLimiterIntegration:
    """
    Drop-in replacement for Agent Zero's rate limiting system
    """
    
    def __init__(self, agent):
        self.agent = agent
        self.enabled = True
        self.learning_enabled = True
        
    async def intelligent_rate_limiter(
        self, 
        model_config: models.ModelConfig, 
        input_text: str, 
        background: bool = False
    ):
        """
        Intelligent replacement for Agent Zero's rate_limiter method
        
        This method:
        1. Gets intelligent limits based on learned patterns
        2. Uses those limits for rate limiting
        3. Records the attempt for future learning
        """
        
        if not self.enabled:
            # Fallback to original behavior
            return await self._original_rate_limiter(model_config, input_text, background)
        
        # Calculate input tokens
        input_tokens = approximate_tokens(input_text)
        
        # Get intelligent rate limiter
        limiter, intelligent_limits = await get_intelligent_rate_limiter_for_model(
            provider=model_config.provider.name.lower(),
            model=model_config.name,
            fallback_requests=model_config.limit_requests or 10,
            fallback_input=model_config.limit_input or 1000,
            fallback_output=model_config.limit_output or 1000
        )
        
        # Enhanced wait callback with intelligence info
        wait_log = None
        
        async def intelligent_wait_callback(msg: str, key: str, total: int, limit: int):
            nonlocal wait_log
            if not wait_log:
                # Enhanced logging with intelligence metrics
                intelligence_info = (
                    f"🧠 Intelligent Limits (confidence: {intelligent_limits.confidence:.1%}, "
                    f"success rate: {intelligent_limits.success_rate:.1%})"
                )
                
                wait_log = self.agent.context.log.log(
                    type="util",
                    update_progress="none",
                    heading=f"{msg} - {intelligence_info}",
                    model=f"{model_config.provider.value}\\{model_config.name}",
                )
            
            # Enhanced message with optimization info
            enhanced_msg = f"{msg} | Efficiency: {intelligent_limits.efficiency_score:.1%}"
            wait_log.update(heading=enhanced_msg, key=key, value=total, limit=limit)
            
            if not background:
                self.agent.context.log.set_progress(enhanced_msg, -1)
        
        # Add the request to the limiter
        limiter.add(input=input_tokens)
        limiter.add(requests=1)
        
        # Wait with intelligent callback
        await limiter.wait(callback=intelligent_wait_callback)
        
        # Return enhanced limiter with intelligence tracking
        return IntelligentLimiterWrapper(limiter, model_config, intelligent_limits, self)
    
    async def _original_rate_limiter(self, model_config: models.ModelConfig, input_text: str, background: bool = False):
        """Fallback to original Agent Zero rate limiting"""
        wait_log = None
        
        # Original Agent Zero rate limiting logic
        input_tokens = approximate_tokens(input_text)
        
        # Create basic rate limiter with original limits
        from python.helpers.rate_limiter import RateLimiter
        limiter = RateLimiter(
            requests=model_config.limit_requests or 10,
            input=model_config.limit_input or 1000,
            output=model_config.limit_output or 1000
        )
        
        async def wait_callback(msg: str, key: str, total: int, limit: int):
            nonlocal wait_log
            if not wait_log:
                wait_log = self.agent.context.log.log(
                    type="util",
                    update_progress="none",
                    heading=msg,
                    model=f"{model_config.provider.value}\\{model_config.name}",
                )
            wait_log.update(heading=msg, key=key, value=total, limit=limit)
            if not background:
                self.agent.context.log.set_progress(msg, -1)
        
        limiter.add(input=input_tokens)
        limiter.add(requests=1)
        await limiter.wait(callback=wait_callback)
        
        return limiter


class IntelligentLimiterWrapper:
    """
    Wrapper that adds intelligence tracking to rate limiter results
    """
    
    def __init__(self, limiter, model_config, intelligent_limits, integration):
        self.limiter = limiter
        self.model_config = model_config
        self.intelligent_limits = intelligent_limits
        self.integration = integration
        self.start_time = time.time()
        
    def add(self, **kwargs):
        """Add to the underlying limiter"""
        return self.limiter.add(**kwargs)
    
    async def wait(self, callback=None):
        """Wait with the underlying limiter"""
        return await self.limiter.wait(callback=callback)
    
    def record_success(self, output_tokens: int = 0, response_time: float = None):
        """Record successful API call for learning"""
        if not self.integration.learning_enabled:
            return
            
        if response_time is None:
            response_time = time.time() - self.start_time
            
        # Record the successful usage
        record_api_usage(
            provider=self.model_config.provider.name.lower(),
            model=self.model_config.name,
            success=True,
            response_time=response_time,
            output_tokens=output_tokens,
            rate_limited=False  # If we got here, we weren't rate limited
        )
        
    def record_failure(self, error_type: str = "unknown", was_rate_limited: bool = False):
        """Record failed API call for learning"""
        if not self.integration.learning_enabled:
            return
            
        response_time = time.time() - self.start_time
        
        # Record the failed usage
        record_api_usage(
            provider=self.model_config.provider.name.lower(),
            model=self.model_config.name,
            success=False,
            response_time=response_time,
            output_tokens=0,
            rate_limited=was_rate_limited,
            error_type=error_type
        )


def integrate_with_agent_zero(agent):
    """
    Integrate intelligent rate limiting with an Agent Zero instance
    
    This function:
    1. Creates an integration instance
    2. Replaces the agent's rate_limiter method
    3. Sets up learning callbacks
    """
    
    # Create integration
    integration = IntelligentRateLimiterIntegration(agent)
    
    # Store original method for fallback
    agent._original_rate_limiter = agent.rate_limiter
    
    # Replace with intelligent version
    agent.rate_limiter = integration.intelligent_rate_limiter
    
    # Store integration for access
    agent._intelligent_rate_limiter = integration
    
    logger.info("🧠 Intelligent rate limiting integrated with Agent Zero")
    
    return integration


def disable_intelligent_rate_limiting(agent):
    """
    Disable intelligent rate limiting and restore original behavior
    """
    if hasattr(agent, '_original_rate_limiter'):
        agent.rate_limiter = agent._original_rate_limiter
        delattr(agent, '_original_rate_limiter')
        
    if hasattr(agent, '_intelligent_rate_limiter'):
        delattr(agent, '_intelligent_rate_limiter')
        
    logger.info("🔄 Restored original Agent Zero rate limiting")


# Decorator for automatic integration
def with_intelligent_rate_limiting(agent_class):
    """
    Class decorator to automatically integrate intelligent rate limiting
    """
    
    original_init = agent_class.__init__
    
    def enhanced_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        integrate_with_agent_zero(self)
    
    agent_class.__init__ = enhanced_init
    return agent_class


# Context manager for temporary intelligent rate limiting
class TemporaryIntelligentRateLimiting:
    """
    Context manager for temporary intelligent rate limiting
    """
    
    def __init__(self, agent, learning_enabled=True):
        self.agent = agent
        self.learning_enabled = learning_enabled
        self.integration = None
        
    async def __aenter__(self):
        self.integration = integrate_with_agent_zero(self.agent)
        self.integration.learning_enabled = self.learning_enabled
        return self.integration
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        disable_intelligent_rate_limiting(self.agent)


# Example usage functions
async def example_basic_integration():
    """
    Example: Basic integration with Agent Zero
    """
    # Assuming you have an Agent Zero instance
    # agent = Agent(...)
    
    # Integrate intelligent rate limiting
    # integration = integrate_with_agent_zero(agent)
    
    # Now all agent API calls will use intelligent rate limiting
    # response = await agent.call_llm("Hello, world!")
    
    print("✅ Basic integration example (commented out - requires Agent Zero instance)")


async def example_temporary_integration():
    """
    Example: Temporary integration using context manager
    """
    # Assuming you have an Agent Zero instance
    # agent = Agent(...)
    
    # Use intelligent rate limiting temporarily
    # async with TemporaryIntelligentRateLimiting(agent) as integration:
    #     response = await agent.call_llm("Hello, world!")
    #     # Intelligent rate limiting is active
    
    # Original rate limiting is restored
    
    print("✅ Temporary integration example (commented out - requires Agent Zero instance)")


async def example_decorator_integration():
    """
    Example: Using decorator for automatic integration
    """
    
    # @with_intelligent_rate_limiting
    # class MyAgent(Agent):
    #     def __init__(self, *args, **kwargs):
    #         super().__init__(*args, **kwargs)
    #         # Intelligent rate limiting is automatically integrated
    
    print("✅ Decorator integration example (commented out - requires Agent Zero class)")


if __name__ == "__main__":
    """
    Demonstration of intelligent rate limiter integration
    """
    
    print("🧠 Agent Zero Intelligent Rate Limiter Integration")
    print("=" * 60)
    
    # Run examples
    asyncio.run(example_basic_integration())
    asyncio.run(example_temporary_integration())
    asyncio.run(example_decorator_integration())
    
    print("\n✅ Integration examples completed")
    print("\nTo use with real Agent Zero:")
    print("1. Import this module")
    print("2. Call integrate_with_agent_zero(your_agent)")
    print("3. All API calls will use intelligent rate limiting")
    print("4. The system learns and optimizes automatically")
