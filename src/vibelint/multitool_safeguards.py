"""
Multitool Safeguards System

Implements judge model safeguards for multitool calls, especially for write and execute
operations. Uses claudiallm as the judge model to evaluate whether multitool operations
are safe to perform.

This addresses security concerns around batch operations that could cause system damage.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SafeguardDecision(Enum):
    """Judge model decision on multitool operation safety."""

    APPROVE = "approve"
    DENY = "deny"
    REQUIRE_HUMAN = "require_human"


@dataclass
class ToolCall:
    """Represents a single tool call in a multitool operation."""

    tool_name: str
    parameters: Dict[str, Any]
    description: str
    risk_level: str = "unknown"


@dataclass
class SafeguardResult:
    """Result from judge model evaluation."""

    decision: SafeguardDecision
    confidence: float
    reasoning: str
    risk_assessment: str
    alternative_suggestions: List[str]
    requires_human_approval: bool = False


class MultitoolSafeguards:
    """
    Judge model safeguards for multitool operations.

    Uses claudiallm to evaluate whether batch operations are safe to perform,
    especially for write/execute operations that could cause system damage.
    """

    # High-risk tool patterns that always require judgment
    HIGH_RISK_TOOLS = {"Write", "MultiEdit", "NotebookEdit", "Bash", "mcp__ide__executeCode"}

    # Tool combinations that are particularly dangerous
    DANGEROUS_COMBINATIONS = [
        {"Write", "Bash"},  # Writing files then executing
        {"MultiEdit", "Bash"},  # Editing multiple files then executing
        {"Write", "Write", "Write"},  # Multiple file writes
    ]

    def __init__(self, llm_client=None):
        """Initialize safeguards with optional LLM client."""
        self.llm_client = llm_client
        self._initialize_judge_model()

    def _initialize_judge_model(self):
        """Initialize the claudiallm judge model."""
        if self.llm_client is None:
            try:
                # Try to import and initialize claudiallm
                # This would be replaced with actual claudiallm initialization
                logger.info("Initializing claudiallm judge model for safeguards")
                # self.llm_client = claudiallm.Client(...)
                logger.warning("claudiallm not available - using fallback safety rules")
            except ImportError:
                logger.warning("claudiallm not available - using fallback safety rules")

    async def evaluate_multitool_operation(
        self, tool_calls: List[ToolCall], context: Optional[str] = None
    ) -> SafeguardResult:
        """
        Evaluate whether a multitool operation is safe to perform.

        Args:
            tool_calls: List of tool calls to evaluate
            context: Optional context about the operation

        Returns:
            SafeguardResult with judge decision and reasoning
        """
        # Quick safety checks first
        if len(tool_calls) == 1:
            # Single tool calls are generally safe, unless high-risk
            if tool_calls[0].tool_name in self.HIGH_RISK_TOOLS:
                return await self._evaluate_single_high_risk_tool(tool_calls[0], context)
            else:
                return SafeguardResult(
                    decision=SafeguardDecision.APPROVE,
                    confidence=0.95,
                    reasoning="Single tool call with low-risk tool",
                    risk_assessment="low",
                    alternative_suggestions=[],
                )

        # Multiple tool calls - requires more careful evaluation
        return await self._evaluate_multitool_batch(tool_calls, context)

    async def _evaluate_single_high_risk_tool(
        self, tool_call: ToolCall, context: Optional[str]
    ) -> SafeguardResult:
        """Evaluate a single high-risk tool call."""
        if self.llm_client:
            return await self._llm_judge_evaluation([tool_call], context, focus="single_high_risk")

        # Fallback rules for high-risk tools
        if tool_call.tool_name == "Bash":
            # Check for dangerous bash commands
            command = tool_call.parameters.get("command", "")
            if self._contains_dangerous_bash_patterns(command):
                return SafeguardResult(
                    decision=SafeguardDecision.DENY,
                    confidence=0.9,
                    reasoning=f"Dangerous bash command detected: {command}",
                    risk_assessment="high",
                    alternative_suggestions=["Break down into smaller, safer commands"],
                )

        return SafeguardResult(
            decision=SafeguardDecision.REQUIRE_HUMAN,
            confidence=0.8,
            reasoning=f"High-risk tool {tool_call.tool_name} requires human approval",
            risk_assessment="medium",
            alternative_suggestions=["Review the operation manually"],
            requires_human_approval=True,
        )

    async def _evaluate_multitool_batch(
        self, tool_calls: List[ToolCall], context: Optional[str]
    ) -> SafeguardResult:
        """Evaluate a batch of multiple tool calls."""
        # Check for dangerous combinations
        tool_names = {tc.tool_name for tc in tool_calls}
        for dangerous_combo in self.DANGEROUS_COMBINATIONS:
            if dangerous_combo.issubset(tool_names):
                if self.llm_client:
                    return await self._llm_judge_evaluation(
                        tool_calls, context, focus="dangerous_combination"
                    )
                else:
                    return SafeguardResult(
                        decision=SafeguardDecision.DENY,
                        confidence=0.95,
                        reasoning=f"Dangerous tool combination detected: {dangerous_combo}",
                        risk_assessment="high",
                        alternative_suggestions=[
                            "Execute tools individually with human review",
                            "Use safer alternatives where possible",
                        ],
                    )

        # Check for too many high-risk operations
        high_risk_count = sum(1 for tc in tool_calls if tc.tool_name in self.HIGH_RISK_TOOLS)
        if high_risk_count >= 3:
            return SafeguardResult(
                decision=SafeguardDecision.REQUIRE_HUMAN,
                confidence=0.85,
                reasoning=f"Too many high-risk operations in batch: {high_risk_count}",
                risk_assessment="high",
                alternative_suggestions=["Break into smaller batches"],
                requires_human_approval=True,
            )

        # Use LLM judge if available
        if self.llm_client:
            return await self._llm_judge_evaluation(tool_calls, context, focus="batch")

        # Fallback: require human approval for complex multitool operations
        return SafeguardResult(
            decision=SafeguardDecision.REQUIRE_HUMAN,
            confidence=0.7,
            reasoning="Complex multitool operation requires human review",
            risk_assessment="medium",
            alternative_suggestions=["Review each tool call individually"],
            requires_human_approval=True,
        )

    async def _llm_judge_evaluation(
        self, tool_calls: List[ToolCall], context: Optional[str], focus: str
    ) -> SafeguardResult:
        """Use claudiallm to evaluate the tool calls."""
        try:
            # Prepare prompt for judge model
            evaluation_prompt = self._build_judge_prompt(tool_calls, context, focus)

            # Call claudiallm (this would be the actual implementation)
            # response = await self.llm_client.complete(evaluation_prompt)
            # For now, simulate LLM response
            response = await self._simulate_llm_judge(evaluation_prompt, tool_calls)

            return self._parse_judge_response(response)

        except Exception as e:
            logger.error(f"Error in LLM judge evaluation: {e}")
            # Fail safe: require human approval on error
            return SafeguardResult(
                decision=SafeguardDecision.REQUIRE_HUMAN,
                confidence=0.5,
                reasoning=f"LLM judge error, requiring human review: {e}",
                risk_assessment="unknown",
                alternative_suggestions=["Manual review recommended"],
                requires_human_approval=True,
            )

    def _build_judge_prompt(
        self, tool_calls: List[ToolCall], context: Optional[str], focus: str
    ) -> str:
        """Build prompt for the judge model."""
        tool_descriptions = []
        for i, tool_call in enumerate(tool_calls, 1):
            tool_descriptions.append(
                f"{i}. {tool_call.tool_name}: {tool_call.description}\n"
                f"   Parameters: {json.dumps(tool_call.parameters, indent=2)}"
            )

        prompt = f"""You are a security judge evaluating whether a multitool operation is safe to execute.

CONTEXT: {context or 'No additional context provided'}

FOCUS: {focus}

TOOL CALLS TO EVALUATE:
{chr(10).join(tool_descriptions)}

Please evaluate this multitool operation and respond with a JSON object containing:
{{
    "decision": "approve" | "deny" | "require_human",
    "confidence": 0.0-1.0,
    "reasoning": "Clear explanation of your decision",
    "risk_assessment": "low" | "medium" | "high",
    "alternative_suggestions": ["list", "of", "alternatives"]
}}

EVALUATION CRITERIA:
- Data safety: Could this damage or corrupt files?
- System safety: Could this harm the system or other processes?
- Scope appropriateness: Is the batch size reasonable?
- Intent clarity: Are the operations clearly related and purposeful?
- Reversibility: Can the effects be easily undone if needed?

HIGH-RISK PATTERNS TO WATCH FOR:
- Multiple file modifications followed by execution
- Operations on system-critical files
- Broad filesystem changes
- Combining write operations with command execution
- Operations without clear safeguards

DECISION GUIDELINES:
- APPROVE: Safe, well-scoped operations with low risk
- DENY: Clearly dangerous or potentially destructive operations
- REQUIRE_HUMAN: Uncertain cases or operations needing human judgment

Respond only with the JSON object, no additional text."""

        return prompt

    async def _simulate_llm_judge(self, prompt: str, tool_calls: List[ToolCall]) -> str:
        """Simulate LLM judge response (replace with actual claudiallm call)."""
        # This is a placeholder simulation - replace with actual claudiallm integration
        await asyncio.sleep(0.1)  # Simulate network delay

        # Simple heuristic-based simulation
        tool_names = [tc.tool_name for tc in tool_calls]
        high_risk_count = sum(1 for name in tool_names if name in self.HIGH_RISK_TOOLS)

        if high_risk_count == 0:
            decision = "approve"
            confidence = 0.9
            risk = "low"
            reasoning = "All tools are low-risk"
        elif high_risk_count == 1 and len(tool_calls) <= 2:
            decision = "require_human"
            confidence = 0.8
            risk = "medium"
            reasoning = "Single high-risk tool in small batch"
        else:
            decision = "deny"
            confidence = 0.85
            risk = "high"
            reasoning = "Multiple high-risk tools or large batch"

        return json.dumps(
            {
                "decision": decision,
                "confidence": confidence,
                "reasoning": reasoning,
                "risk_assessment": risk,
                "alternative_suggestions": [
                    "Break into smaller operations",
                    "Add explicit safeguards",
                ],
            }
        )

    def _parse_judge_response(self, response: str) -> SafeguardResult:
        """Parse the judge model JSON response."""
        try:
            data = json.loads(response)
            return SafeguardResult(
                decision=SafeguardDecision(data["decision"]),
                confidence=float(data["confidence"]),
                reasoning=data["reasoning"],
                risk_assessment=data["risk_assessment"],
                alternative_suggestions=data.get("alternative_suggestions", []),
                requires_human_approval=(data["decision"] == "require_human"),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Error parsing judge response: {e}")
            # Fail safe
            return SafeguardResult(
                decision=SafeguardDecision.REQUIRE_HUMAN,
                confidence=0.5,
                reasoning=f"Failed to parse judge response: {e}",
                risk_assessment="unknown",
                alternative_suggestions=["Manual review required"],
                requires_human_approval=True,
            )

    def _contains_dangerous_bash_patterns(self, command: str) -> bool:
        """Check if bash command contains dangerous patterns."""
        dangerous_patterns = [
            "rm -rf",
            "sudo rm",
            "format",
            "mkfs",
            "dd if=",
            ":(){ :|:& };:",
            "chmod -R 777",
            "chown -R",
            "curl | sh",
            "wget | sh",
            ">{PATH}",
            "cat /dev/zero",
            "python -c 'import os;os.system'",
        ]

        command_lower = command.lower()
        return any(pattern.lower() in command_lower for pattern in dangerous_patterns)


# Global safeguards instance
_safeguards_instance: Optional[MultitoolSafeguards] = None


def get_safeguards() -> MultitoolSafeguards:
    """Get or create the global safeguards instance."""
    global _safeguards_instance
    if _safeguards_instance is None:
        _safeguards_instance = MultitoolSafeguards()
    return _safeguards_instance


async def evaluate_multitool_safety(
    tool_calls: List[Dict[str, Any]], context: Optional[str] = None
) -> SafeguardResult:
    """
    Convenience function to evaluate multitool operation safety.

    Args:
        tool_calls: List of tool call dictionaries
        context: Optional context about the operation

    Returns:
        SafeguardResult with safety evaluation
    """
    safeguards = get_safeguards()

    # Convert dict tool calls to ToolCall objects
    tool_call_objects = []
    for tool_dict in tool_calls:
        tool_call = ToolCall(
            tool_name=tool_dict.get("tool_name", "unknown"),
            parameters=tool_dict.get("parameters", {}),
            description=tool_dict.get(
                "description", f"Call to {tool_dict.get('tool_name', 'unknown')}"
            ),
        )
        tool_call_objects.append(tool_call)

    return await safeguards.evaluate_multitool_operation(tool_call_objects, context)


# Decorator for protecting multitool operations
def require_safeguard_approval(context: str = None):
    """
    Decorator to require safeguard approval for multitool operations.

    Usage:
        @require_safeguard_approval("File modification operation")
        async def modify_files(tool_calls):
            # Implementation
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract tool_calls from args/kwargs
            tool_calls = []
            if args and isinstance(args[0], list):
                tool_calls = args[0]
            elif "tool_calls" in kwargs:
                tool_calls = kwargs["tool_calls"]

            if len(tool_calls) > 1:
                result = await evaluate_multitool_safety(tool_calls, context)

                if result.decision == SafeguardDecision.DENY:
                    raise PermissionError(
                        f"Multitool operation denied by safeguards: {result.reasoning}"
                    )
                elif result.decision == SafeguardDecision.REQUIRE_HUMAN:
                    # In a real implementation, this would prompt for human approval
                    logger.warning(
                        f"Multitool operation requires human approval: {result.reasoning}"
                    )

            return await func(*args, **kwargs)

        return wrapper

    return decorator
