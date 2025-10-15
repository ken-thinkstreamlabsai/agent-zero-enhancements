#!/usr/bin/env python3
"""
Intelligent Rate Limiter for Agent Zero
Self-tuning API call management using ecosystem intelligence
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Callable, Awaitable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import models
from python.helpers.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class APILimitType(Enum):
    REQUESTS_PER_MINUTE = "requests_per_minute"
    TOKENS_PER_MINUTE = "tokens_per_minute"
    TOKENS_PER_DAY = "tokens_per_day"
    CONCURRENT_REQUESTS = "concurrent_requests"


@dataclass
class APIProviderLimits:
    """Known API provider limits and characteristics"""
    provider: str
    model: str
    
    # Known limits (from API documentation)
    requests_per_minute: int = 0
    tokens_per_minute: int = 0
    tokens_per_day: int = 0
    concurrent_requests: int = 0
    
    # Observed characteristics
    typical_response_time: float = 1.0
    burst_tolerance: float = 0.8  # How close to limit we can safely go
    backoff_multiplier: float = 1.5
    
    # Cost information
    cost_per_1k_input_tokens: float = 0.0
    cost_per_1k_output_tokens: float = 0.0


@dataclass
class APIUsageEvent:
    """Record of an API call for learning"""
    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    response_time: float
    success: bool
    error_type: Optional[str] = None
    rate_limited: bool = False
    cost: float = 0.0


@dataclass
class IntelligentLimits:
    """Dynamically calculated limits"""
    requests_per_minute: int
    input_tokens_per_minute: int
    output_tokens_per_minute: int
    confidence: float  # 0.0 to 1.0
    last_updated: datetime
    success_rate: float
    average_response_time: float
    
    # Optimization metrics
    efficiency_score: float  # How well we're using available capacity
    cost_efficiency: float   # Cost per successful operation
    
    def to_dict(self) -> Dict:
        return {
            'requests_per_minute': self.requests_per_minute,
            'input_tokens_per_minute': self.input_tokens_per_minute,
            'output_tokens_per_minute': self.output_tokens_per_minute,
            'confidence': self.confidence,
            'last_updated': self.last_updated.isoformat(),
            'success_rate': self.success_rate,
            'average_response_time': self.average_response_time,
            'efficiency_score': self.efficiency_score,
            'cost_efficiency': self.cost_efficiency
        }


class IntelligentRateLimiter:
    """
    Intelligent rate limiter that learns and adapts API call patterns
    """
    
    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.usage_history: List[APIUsageEvent] = []
        self.provider_limits: Dict[str, APIProviderLimits] = {}
        self.intelligent_limits: Dict[str, IntelligentLimits] = {}
        self.learning_enabled = True
        self.optimization_enabled = True
        
        # Learning parameters
        self.min_samples_for_learning = 10
        self.learning_window_hours = 24
        self.confidence_threshold = 0.7
        
        # Initialize known provider limits
        self._initialize_provider_limits()
        
        # Rate limiter instances
        self.rate_limiters: Dict[str, RateLimiter] = {}
        
        logger.info(f"Initialized IntelligentRateLimiter for agent {agent_id}")
    
    def _initialize_provider_limits(self):
        """Initialize known API provider limits"""
        
        # OpenAI GPT-4 limits (as of 2024)
        self.provider_limits["openai_gpt-4"] = APIProviderLimits(
            provider="openai",
            model="gpt-4",
            requests_per_minute=500,
            tokens_per_minute=30000,
            tokens_per_day=300000,
            concurrent_requests=10,
            typical_response_time=2.0,
            cost_per_1k_input_tokens=0.03,
            cost_per_1k_output_tokens=0.06
        )
        
        # OpenAI GPT-3.5 Turbo limits
        self.provider_limits["openai_gpt-3.5-turbo"] = APIProviderLimits(
            provider="openai",
            model="gpt-3.5-turbo",
            requests_per_minute=3500,
            tokens_per_minute=90000,
            tokens_per_day=2000000,
            concurrent_requests=20,
            typical_response_time=1.0,
            cost_per_1k_input_tokens=0.001,
            cost_per_1k_output_tokens=0.002
        )
        
        # Anthropic Claude limits
        self.provider_limits["anthropic_claude-3-sonnet"] = APIProviderLimits(
            provider="anthropic",
            model="claude-3-sonnet",
            requests_per_minute=1000,
            tokens_per_minute=80000,
            tokens_per_day=1000000,
            concurrent_requests=5,
            typical_response_time=1.5,
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015
        )
        
        # Add more providers as needed...
        logger.info(f"Initialized {len(self.provider_limits)} provider limit profiles")
    
    def get_model_key(self, provider: str, model: str) -> str:
        """Generate consistent key for provider/model combination"""
        return f"{provider}_{model}".lower().replace("-", "_").replace(".", "_")
    
    async def record_api_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        response_time: float,
        success: bool,
        error_type: Optional[str] = None,
        rate_limited: bool = False
    ):
        """Record an API call for learning"""
        
        model_key = self.get_model_key(provider, model)
        
        # Calculate cost if we have pricing info
        cost = 0.0
        if model_key in self.provider_limits:
            limits = self.provider_limits[model_key]
            cost = (
                (input_tokens / 1000) * limits.cost_per_1k_input_tokens +
                (output_tokens / 1000) * limits.cost_per_1k_output_tokens
            )
        
        event = APIUsageEvent(
            timestamp=datetime.now(timezone.utc),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            response_time=response_time,
            success=success,
            error_type=error_type,
            rate_limited=rate_limited,
            cost=cost
        )
        
        self.usage_history.append(event)
        
        # Trigger learning if we have enough samples
        if self.learning_enabled and len(self.usage_history) % 5 == 0:
            await self._update_intelligent_limits(model_key)
        
        logger.debug(f"Recorded API call: {provider}/{model}, success={success}, tokens={input_tokens}+{output_tokens}")
    
    async def _update_intelligent_limits(self, model_key: str):
        """Update intelligent limits based on usage history"""
        
        # Get recent usage for this model
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.learning_window_hours)
        recent_usage = [
            event for event in self.usage_history
            if (self.get_model_key(event.provider, event.model) == model_key and
                event.timestamp > cutoff_time)
        ]
        
        if len(recent_usage) < self.min_samples_for_learning:
            logger.debug(f"Not enough samples for {model_key}: {len(recent_usage)}")
            return
        
        # Calculate success rate
        successful_calls = [event for event in recent_usage if event.success]
        success_rate = len(successful_calls) / len(recent_usage)
        
        # Calculate average response time
        avg_response_time = statistics.mean([event.response_time for event in recent_usage])
        
        # Analyze rate limiting patterns
        rate_limited_calls = [event for event in recent_usage if event.rate_limited]
        rate_limit_frequency = len(rate_limited_calls) / len(recent_usage)
        
        # Get known limits for this model
        provider_limits = self.provider_limits.get(model_key)
        
        if provider_limits:
            # Calculate optimal limits based on success rate and known limits
            base_requests = provider_limits.requests_per_minute
            base_tokens = provider_limits.tokens_per_minute
            
            # Adjust based on observed success rate
            if success_rate < 0.9:  # If success rate is low, be more conservative
                adjustment_factor = 0.7
            elif success_rate > 0.95 and rate_limit_frequency < 0.1:  # If doing well, be more aggressive
                adjustment_factor = 0.9
            else:
                adjustment_factor = 0.8  # Default conservative approach
            
            # Calculate intelligent limits
            intelligent_requests = int(base_requests * adjustment_factor)
            intelligent_input_tokens = int(base_tokens * adjustment_factor * 0.6)  # Assume 60% input
            intelligent_output_tokens = int(base_tokens * adjustment_factor * 0.4)  # Assume 40% output
            
        else:
            # No known limits, learn from observed patterns
            # This is more conservative but learns from actual usage
            
            # Calculate tokens per minute from recent usage
            if len(recent_usage) > 0:
                # Group by minute and calculate averages
                minute_groups = {}
                for event in recent_usage:
                    minute_key = event.timestamp.replace(second=0, microsecond=0)
                    if minute_key not in minute_groups:
                        minute_groups[minute_key] = []
                    minute_groups[minute_key].append(event)
                
                if minute_groups:
                    avg_requests_per_minute = statistics.mean([len(events) for events in minute_groups.values()])
                    avg_input_tokens_per_minute = statistics.mean([
                        sum(event.input_tokens for event in events) for events in minute_groups.values()
                    ])
                    avg_output_tokens_per_minute = statistics.mean([
                        sum(event.output_tokens for event in events) for events in minute_groups.values()
                    ])
                    
                    # Be conservative with learned limits
                    intelligent_requests = max(1, int(avg_requests_per_minute * 0.8))
                    intelligent_input_tokens = max(100, int(avg_input_tokens_per_minute * 0.8))
                    intelligent_output_tokens = max(100, int(avg_output_tokens_per_minute * 0.8))
                else:
                    # Fallback defaults
                    intelligent_requests = 10
                    intelligent_input_tokens = 1000
                    intelligent_output_tokens = 1000
            else:
                # Fallback defaults
                intelligent_requests = 10
                intelligent_input_tokens = 1000
                intelligent_output_tokens = 1000
        
        # Calculate confidence based on sample size and consistency
        confidence = min(1.0, len(recent_usage) / 100.0)  # More samples = higher confidence
        if success_rate < 0.8:
            confidence *= 0.5  # Lower confidence if success rate is poor
        
        # Calculate efficiency metrics
        if provider_limits:
            efficiency_score = (intelligent_requests / provider_limits.requests_per_minute) * success_rate
        else:
            efficiency_score = success_rate  # Without known limits, use success rate as proxy
        
        # Calculate cost efficiency
        total_cost = sum(event.cost for event in successful_calls)
        cost_efficiency = len(successful_calls) / max(total_cost, 0.001)  # Operations per dollar
        
        # Create intelligent limits
        intelligent_limits = IntelligentLimits(
            requests_per_minute=intelligent_requests,
            input_tokens_per_minute=intelligent_input_tokens,
            output_tokens_per_minute=intelligent_output_tokens,
            confidence=confidence,
            last_updated=datetime.now(timezone.utc),
            success_rate=success_rate,
            average_response_time=avg_response_time,
            efficiency_score=efficiency_score,
            cost_efficiency=cost_efficiency
        )
        
        self.intelligent_limits[model_key] = intelligent_limits
        
        logger.info(f"Updated intelligent limits for {model_key}: "
                   f"requests={intelligent_requests}/min, "
                   f"input_tokens={intelligent_input_tokens}/min, "
                   f"confidence={confidence:.2f}, "
                   f"success_rate={success_rate:.2f}")
    
    async def get_intelligent_rate_limiter(
        self,
        provider: str,
        model: str,
        fallback_requests: int = 10,
        fallback_input: int = 1000,
        fallback_output: int = 1000
    ) -> Tuple[RateLimiter, IntelligentLimits]:
        """Get an intelligent rate limiter for the specified model"""
        
        model_key = self.get_model_key(provider, model)
        
        # Get intelligent limits or create defaults
        if model_key in self.intelligent_limits:
            limits = self.intelligent_limits[model_key]
            
            # Use intelligent limits if confidence is high enough
            if limits.confidence >= self.confidence_threshold:
                requests_limit = limits.requests_per_minute
                input_limit = limits.input_tokens_per_minute
                output_limit = limits.output_tokens_per_minute
                logger.info(f"Using intelligent limits for {model_key} (confidence: {limits.confidence:.2f})")
            else:
                # Use conservative defaults if confidence is low
                requests_limit = fallback_requests
                input_limit = fallback_input
                output_limit = fallback_output
                logger.info(f"Using fallback limits for {model_key} (low confidence: {limits.confidence:.2f})")
        else:
            # No intelligent limits yet, check if we have provider defaults
            if model_key in self.provider_limits:
                provider_limits = self.provider_limits[model_key]
                requests_limit = int(provider_limits.requests_per_minute * 0.8)  # Conservative
                input_limit = int(provider_limits.tokens_per_minute * 0.5)  # Conservative
                output_limit = int(provider_limits.tokens_per_minute * 0.3)  # Conservative
                logger.info(f"Using provider defaults for {model_key}")
            else:
                # Use fallback defaults
                requests_limit = fallback_requests
                input_limit = fallback_input
                output_limit = fallback_output
                logger.info(f"Using fallback defaults for {model_key}")
            
            # Create initial intelligent limits
            limits = IntelligentLimits(
                requests_per_minute=requests_limit,
                input_tokens_per_minute=input_limit,
                output_tokens_per_minute=output_limit,
                confidence=0.1,  # Low initial confidence
                last_updated=datetime.now(timezone.utc),
                success_rate=1.0,  # Optimistic initial assumption
                average_response_time=2.0,  # Conservative assumption
                efficiency_score=0.5,
                cost_efficiency=1.0
            )
            self.intelligent_limits[model_key] = limits
        
        # Create or get rate limiter
        if model_key not in self.rate_limiters:
            self.rate_limiters[model_key] = RateLimiter(
                seconds=60,
                requests=requests_limit,
                input=input_limit,
                output=output_limit
            )
        else:
            # Update existing rate limiter with new limits
            limiter = self.rate_limiters[model_key]
            limiter.limits["requests"] = requests_limit
            limiter.limits["input"] = input_limit
            limiter.limits["output"] = output_limit
        
        return self.rate_limiters[model_key], limits
    
    async def optimize_for_cost(self, model_key: str) -> bool:
        """Optimize limits to minimize cost while maintaining performance"""
        if model_key not in self.intelligent_limits:
            return False
        
        limits = self.intelligent_limits[model_key]
        
        # Get recent usage
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=1)  # Last hour
        recent_usage = [
            event for event in self.usage_history
            if (self.get_model_key(event.provider, event.model) == model_key and
                event.timestamp > cutoff_time)
        ]
        
        if len(recent_usage) < 5:
            return False
        
        # If success rate is high and we're not hitting limits, we can be more aggressive
        success_rate = len([e for e in recent_usage if e.success]) / len(recent_usage)
        rate_limited_rate = len([e for e in recent_usage if e.rate_limited]) / len(recent_usage)
        
        if success_rate > 0.95 and rate_limited_rate < 0.05:
            # Increase limits by 10%
            limits.requests_per_minute = int(limits.requests_per_minute * 1.1)
            limits.input_tokens_per_minute = int(limits.input_tokens_per_minute * 1.1)
            limits.output_tokens_per_minute = int(limits.output_tokens_per_minute * 1.1)
            limits.last_updated = datetime.now(timezone.utc)
            
            logger.info(f"Optimized limits for {model_key} - increased by 10%")
            return True
        
        elif success_rate < 0.9 or rate_limited_rate > 0.1:
            # Decrease limits by 20%
            limits.requests_per_minute = int(limits.requests_per_minute * 0.8)
            limits.input_tokens_per_minute = int(limits.input_tokens_per_minute * 0.8)
            limits.output_tokens_per_minute = int(limits.output_tokens_per_minute * 0.8)
            limits.last_updated = datetime.now(timezone.utc)
            
            logger.info(f"Optimized limits for {model_key} - decreased by 20%")
            return True
        
        return False
    
    def get_optimization_recommendations(self) -> Dict[str, Dict]:
        """Get optimization recommendations for all models"""
        recommendations = {}
        
        for model_key, limits in self.intelligent_limits.items():
            rec = {
                'current_limits': limits.to_dict(),
                'recommendations': []
            }
            
            # Analyze performance
            if limits.success_rate < 0.9:
                rec['recommendations'].append({
                    'type': 'reduce_limits',
                    'reason': f'Low success rate ({limits.success_rate:.2f})',
                    'suggested_reduction': '20%'
                })
            
            if limits.efficiency_score < 0.5:
                rec['recommendations'].append({
                    'type': 'optimize_usage',
                    'reason': f'Low efficiency score ({limits.efficiency_score:.2f})',
                    'suggestion': 'Consider batching requests or using different model'
                })
            
            if limits.average_response_time > 5.0:
                rec['recommendations'].append({
                    'type': 'reduce_load',
                    'reason': f'High response time ({limits.average_response_time:.2f}s)',
                    'suggestion': 'Reduce concurrent requests'
                })
            
            if limits.confidence < 0.5:
                rec['recommendations'].append({
                    'type': 'gather_data',
                    'reason': f'Low confidence ({limits.confidence:.2f})',
                    'suggestion': 'Need more usage data for better optimization'
                })
            
            recommendations[model_key] = rec
        
        return recommendations
    
    def export_intelligence_data(self) -> Dict:
        """Export intelligence data for analysis or backup"""
        return {
            'agent_id': self.agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'intelligent_limits': {
                key: limits.to_dict() for key, limits in self.intelligent_limits.items()
            },
            'usage_history_summary': {
                'total_calls': len(self.usage_history),
                'success_rate': len([e for e in self.usage_history if e.success]) / max(len(self.usage_history), 1),
                'total_cost': sum(e.cost for e in self.usage_history),
                'models_used': list(set(self.get_model_key(e.provider, e.model) for e in self.usage_history))
            },
            'optimization_recommendations': self.get_optimization_recommendations()
        }


# Global intelligent rate limiter instance
intelligent_rate_limiter = IntelligentRateLimiter()


async def get_intelligent_rate_limiter_for_model(
    provider: str,
    model: str,
    fallback_requests: int = 10,
    fallback_input: int = 1000,
    fallback_output: int = 1000
) -> Tuple[RateLimiter, IntelligentLimits]:
    """
    Get an intelligent rate limiter for the specified model
    This is the main interface for Agent Zero integration
    """
    return await intelligent_rate_limiter.get_intelligent_rate_limiter(
        provider, model, fallback_requests, fallback_input, fallback_output
    )


async def record_api_usage(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    response_time: float,
    success: bool,
    error_type: Optional[str] = None,
    rate_limited: bool = False
):
    """
    Record API usage for learning
    This should be called after every API call
    """
    await intelligent_rate_limiter.record_api_call(
        provider, model, input_tokens, output_tokens, 
        response_time, success, error_type, rate_limited
    )