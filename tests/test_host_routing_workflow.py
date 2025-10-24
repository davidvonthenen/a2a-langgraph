"""Workflow regression tests for the host routing agent."""

from __future__ import annotations

import unittest

from a2a.types import Message, Part, Role, Task, TaskState, TaskStatus, TextPart

from host_agent.policy_manager import TravelPolicyManager
from host_agent.routing_agent import RoutingAgent


class StubRoutingAgent(RoutingAgent):
    """Routing agent variant that returns canned responses for remote calls."""

    def __init__(self, responses: dict[str, list[str]]) -> None:
        super().__init__(policy_manager=TravelPolicyManager())
        self._weather_agent_name = "Weather Specialist"
        self._airbnb_agent_name = "Airbnb Specialist"
        self._responses: dict[str, list[str]] = {
            name: list(outputs) for name, outputs in responses.items()
        }

    async def _send_message(  # type: ignore[override]
        self, agent_name: str, task: str, session_id: str
    ) -> Task:
        if agent_name not in self._responses:
            raise AssertionError(f"No stubbed responses for agent {agent_name!r}")
        queue = self._responses[agent_name]
        if not queue:
            raise AssertionError(
                f"Stubbed responses for agent {agent_name!r} were exhausted"
            )

        output = queue.pop(0)
        context_key = (session_id, agent_name)
        context_id = f"ctx-{agent_name}"
        self._session_context_ids[context_key] = context_id

        message = Message(
            messageId=f"msg-{agent_name}-{len(queue)}",
            role=Role.agent,
            parts=[Part(root=TextPart(text=output))],
        )
        return Task(
            id=f"task-{agent_name}-{len(queue)}",
            contextId=context_id,
            status=TaskStatus(state=TaskState.COMPLETED, message=message),
        )


class RoutingAgentWorkflowTest(unittest.IsolatedAsyncioTestCase):
    """Validate policy-driven coordination between weather and Airbnb flows."""

    async def test_airbnb_blocked_for_rainy_seattle(self) -> None:
        agent = StubRoutingAgent(
            {
                "Weather Specialist": [
                    "Severe storm warning with dangerous flooding expected for Seattle.",
                ]
            }
        )

        responses = await agent.handle_user_message(
            "I'd like to book an Airbnb in Seattle, WA this weekend.",
            session_id="session-block",
        )

        self.assertFalse(
            any("Airbnb ideas" in chunk for chunk in responses),
            "Hazardous weather should block rental suggestions.",
        )
        self.assertIn("hazardous conditions", responses[-1])

    async def test_airbnb_allowed_for_sunny_long_beach(self) -> None:
        agent = StubRoutingAgent(
            {
                "Weather Specialist": [
                    "Warm sunshine and gentle breezes over Long Beach, CA all weekend.",
                ],
                "Airbnb Specialist": [
                    "Consider the Ocean Breeze loft a short walk from the Long Beach pier.",
                ],
            }
        )

        responses = await agent.handle_user_message(
            "Help me secure an Airbnb in Long Beach, CA.",
            session_id="session-allow",
        )

        self.assertTrue(
            any("Airbnb ideas" in chunk for chunk in responses),
            "Clear weather should allow the Airbnb specialist to respond.",
        )
        self.assertIn("Here are some rental ideas", responses[-1])

    async def test_session_recovers_after_blocked_destination(self) -> None:
        agent = StubRoutingAgent(
            {
                "Weather Specialist": [
                    "Severe storm warning with dangerous flooding expected for Seattle.",
                    "Bright sunshine and calm skies over Long Beach, CA.",
                ],
                "Airbnb Specialist": [
                    "Try the Sunny Surf bungalow near Long Beach's waterfront.",
                ],
            }
        )

        seattle_responses = await agent.handle_user_message(
            "Book an Airbnb in Seattle, WA for next Friday.",
            session_id="session-switch",
        )
        self.assertIn("hazardous conditions", seattle_responses[-1])

        long_beach_responses = await agent.handle_user_message(
            "Okay, how about Long Beach, CA instead?",
            session_id="session-switch",
        )
        self.assertTrue(
            any("Airbnb ideas" in chunk for chunk in long_beach_responses),
            "Once a safe forecast arrives, the agent should surface rentals.",
        )
        self.assertIn("Here are some rental ideas", long_beach_responses[-1])


if __name__ == "__main__":
    unittest.main()
