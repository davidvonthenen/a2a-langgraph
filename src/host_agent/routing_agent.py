"""Routing agent orchestrated by a LangGraph policy."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, TypedDict

import httpx
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    DataPart,
    FilePart,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TextPart,
)
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from typing_extensions import NotRequired

from .policy_manager import PolicyClassification, TravelPolicyManager
from .remote_agent_connection import RemoteAgentConnections

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Remote card discovery may involve slow remote agents. Give the HTTP client a
# generous timeout so initialization succeeds in those cases.
DEFAULT_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=30.0)


class HostGraphState(TypedDict, total=False):
    """State container passed between LangGraph nodes."""

    user_message: str
    session_id: str
    response_chunks: list[str]
    policy_notes: list[str]
    need_weather: bool
    need_airbnb: bool
    location_hint: NotRequired[str | None]
    weather_output: NotRequired[str | None]
    airbnb_output: NotRequired[str | None]
    policy_route: NotRequired[str]
    final_decision: NotRequired[str | None]


class RoutingAgent:
    """Delegates user requests to remote agents using a LangGraph policy."""

    def __init__(
        self,
        *,
        policy_manager: TravelPolicyManager | None = None,
    ) -> None:
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self._session_history: dict[str, list[dict[str, str]]] = {}
        self._session_context_ids: dict[tuple[str, str], str] = {}
        self._policy_manager = policy_manager or TravelPolicyManager()
        self._graph = self._build_graph()
        self._weather_agent_name: str | None = None
        self._airbnb_agent_name: str | None = None
        logger.info("RoutingAgent initialized with LangGraph policy orchestrator")

    async def _async_init_components(
        self, remote_agent_addresses: list[str]
    ) -> None:
        async with httpx.AsyncClient(timeout=DEFAULT_HTTP_TIMEOUT) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                except Exception as exc:  # pragma: no cover - initialization logging
                    logger.error(
                        "Failed to load agent card from %s: %s", address, exc
                    )
                    continue

                remote_connection = RemoteAgentConnections(
                    agent_card=card, agent_url=address
                )
                self.remote_agent_connections[card.name] = remote_connection
                self.cards[card.name] = card
                self._maybe_track_specialist(card)

    @classmethod
    async def create(
        cls,
        remote_agent_addresses: list[str],
    ) -> RoutingAgent:
        instance = cls()
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def list_remote_agents(self) -> list[dict[str, str]]:
        agent_info = []
        for card in self.cards.values():
            agent_info.append(
                {"name": card.name, "description": card.description or ""}
            )
        return agent_info

    async def handle_user_message(
        self, message: str, session_id: str
    ) -> list[str]:
        history = self._history_for_session(session_id)
        history.append({"role": "user", "content": message})

        initial_state: HostGraphState = {
            "user_message": message,
            "session_id": session_id,
        }
        final_state = await self._graph.ainvoke(initial_state)
        responses = final_state.get("response_chunks", [])

        if not responses:
            fallback = "I'm not sure how to help with that just yet."
            history.append({"role": "assistant", "content": fallback})
            return [fallback]

        history.extend({"role": "assistant", "content": text} for text in responses)
        return responses

    def _build_graph(self):
        graph = StateGraph(HostGraphState)

        graph.add_node("classify_request", self._classify_request)
        graph.add_node("evaluate_policy", self._evaluate_policy)
        graph.add_node("fetch_weather", self._fetch_weather)
        graph.add_node("fetch_airbnb", self._fetch_airbnb)
        graph.add_node("compose_response", self._compose_response)

        graph.set_entry_point("classify_request")
        graph.add_edge("classify_request", "evaluate_policy")
        graph.add_conditional_edges(
            "evaluate_policy",
            self._route_policy,
            {
                "fetch_weather": "fetch_weather",
                "fetch_airbnb": "fetch_airbnb",
                "deny_airbnb": "compose_response",
                "respond": "compose_response",
            },
        )
        graph.add_edge("fetch_weather", "evaluate_policy")
        graph.add_edge("fetch_airbnb", "compose_response")
        graph.add_edge("compose_response", END)

        return graph.compile()

    async def _send_message(
        self, agent_name: str, task: str, session_id: str
    ) -> Task | None:
        connection = self.remote_agent_connections.get(agent_name)
        if connection is None:
            logger.error("Unknown agent requested: %s", agent_name)
            return None

        context_key = (session_id, agent_name)
        context_id = self._session_context_ids.get(context_key)
        message_id = uuid.uuid4().hex
        payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
            }
        }
        if context_id:
            payload["message"]["contextId"] = context_id

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await connection.send_message(
            message_request=message_request
        )

        if not isinstance(send_response.root, SendMessageSuccessResponse):
            logger.error("Received non-success response from %s", agent_name)
            return None

        result = send_response.root.result
        if not isinstance(result, Task):
            logger.error("Received non-task response from %s", agent_name)
            return None

        self._session_context_ids[context_key] = result.context_id
        return result

    def _extract_task_output(self, task: Task | None) -> str:
        if task is None or task.status is None or task.status.message is None:
            return ""
        message = task.status.message
        if not message.parts:
            return ""
        parts_text = [self._part_to_text(part) for part in message.parts]
        return "\n".join(filter(None, parts_text))

    def _part_to_text(self, part: Part) -> str:
        root = part.root
        if isinstance(root, TextPart):
            return root.text
        if isinstance(root, DataPart):
            return json.dumps(root.data, indent=2)
        if isinstance(root, FilePart):
            return f"Received file content ({root.file.mime_type or 'unknown mime type'})."
        return ""


    def _maybe_track_specialist(self, card: AgentCard) -> None:
        """Record which remote cards correspond to weather or rental experts."""

        lowered_name = card.name.lower()
        lowered_description = (card.description or "").lower()
        if self._weather_agent_name is None and (
            "weather" in lowered_name or "weather" in lowered_description
        ):
            self._weather_agent_name = card.name
        if self._airbnb_agent_name is None and (
            "airbnb" in lowered_name
            or "rental" in lowered_name
            or "airbnb" in lowered_description
            or "rental" in lowered_description
        ):
            self._airbnb_agent_name = card.name

    def _history_for_session(self, session_id: str) -> list[dict[str, str]]:
        return self._session_history.setdefault(session_id, [])

    def _classify_request(self, state: HostGraphState) -> HostGraphState:
        classification: PolicyClassification = self._policy_manager.classify_request(
            state["user_message"]
        )
        response_chunks = []
        policy_notes = []
        if classification.note:
            policy_notes.append(classification.note)
        if classification.need_rentals:
            response_chunks.append(
                "Policy check: I'll review the weather before sharing Airbnb ideas."
            )
        elif classification.need_weather:
            response_chunks.append(
                "Policy check: looping in the weather specialist for you."
            )

        return {
            "session_id": state["session_id"],
            "user_message": state["user_message"],
            "response_chunks": response_chunks,
            "policy_notes": policy_notes,
            "need_weather": classification.need_weather,
            "need_airbnb": classification.need_rentals,
            "location_hint": classification.location_hint,
        }

    def _evaluate_policy(self, state: HostGraphState) -> HostGraphState:
        policy_notes = list(state.get("policy_notes", []))

        if state.get("need_weather") and not state.get("weather_output"):
            if not self._weather_agent_name:
                policy_notes.append(
                    "Policy fallback: weather specialist unavailable; responding directly."
                )
                return {
                    **state,
                    "policy_notes": policy_notes,
                    "policy_route": "respond",
                    "final_decision": state.get("final_decision"),
                }
            policy_notes.append(
                "Policy: awaiting weather data before continuing with the plan."
            )
            return {
                **state,
                "policy_notes": policy_notes,
                "policy_route": "fetch_weather",
            }

        if state.get("need_airbnb"):
            weather_output = state.get("weather_output") or ""
            if not weather_output:
                policy_notes.append(
                    "Policy: weather result missing, re-requesting from specialist."
                )
                return {
                    **state,
                    "policy_notes": policy_notes,
                    "policy_route": "fetch_weather",
                }
            if self._policy_manager.should_block_rentals(weather_output):
                policy_notes.append(
                    "Policy: hazardous conditions detected, pausing Airbnb guidance."
                )
                return {
                    **state,
                    "policy_notes": policy_notes,
                    "policy_route": "deny_airbnb",
                    "final_decision": "deny_airbnb",
                }
            if not self._airbnb_agent_name:
                policy_notes.append(
                    "Policy fallback: Airbnb specialist unavailable, sharing weather only."
                )
                return {
                    **state,
                    "policy_notes": policy_notes,
                    "policy_route": "respond",
                    "final_decision": "allow_airbnb",
                }
            policy_notes.append(
                "Policy: weather looks acceptable, gathering Airbnb suggestions."
            )
            return {
                **state,
                "policy_notes": policy_notes,
                "policy_route": "fetch_airbnb",
                "final_decision": "allow_airbnb",
            }

        return {
            **state,
            "policy_notes": policy_notes,
            "policy_route": "respond",
            "final_decision": state.get("final_decision"),
        }

    def _route_policy(self, state: HostGraphState) -> str:
        return state.get("policy_route", "respond")

    async def _fetch_weather(self, state: HostGraphState) -> HostGraphState:
        responses = list(state.get("response_chunks", []))
        session_id = state["session_id"]
        agent_name = self._weather_agent_name
        if not agent_name:
            responses.append("Weather specialist is offline right now.")
            return {**state, "response_chunks": responses, "weather_output": ""}

        location = state.get("location_hint")
        location_text = f" for {location}" if location else ""
        task_prompt = (
            "You are assisting a travel policy review. "
            "Provide a concise forecast highlighting any safety risks.\n"
            f"User request:{os.linesep}{state['user_message']}"
        )
        task = await self._send_message(agent_name, task_prompt, session_id)
        weather_output = self._extract_task_output(task)

        if weather_output:
            responses.append(
                f"Weather outlook{location_text}:\n{weather_output.strip()}"
            )
        else:
            responses.append(
                f"I could not retrieve a weather update{location_text or ''}."
            )

        return {
            **state,
            "response_chunks": responses,
            "weather_output": weather_output,
        }

    async def _fetch_airbnb(self, state: HostGraphState) -> HostGraphState:
        responses = list(state.get("response_chunks", []))
        session_id = state["session_id"]
        agent_name = self._airbnb_agent_name
        if not agent_name:
            responses.append("Airbnb specialist is offline right now.")
            return {**state, "response_chunks": responses, "airbnb_output": ""}

        task_prompt = (
            "The host assistant cleared the weather policy check and now needs "
            "fictional Airbnb ideas. Use the weather summary below to tailor your reply.\n"
            f"Weather summary:{os.linesep}{state.get('weather_output', 'Not available')}\n"
            f"User request:{os.linesep}{state['user_message']}"
        )
        task = await self._send_message(agent_name, task_prompt, session_id)
        airbnb_output = self._extract_task_output(task)

        if airbnb_output:
            responses.append(f"Airbnb ideas:\n{airbnb_output.strip()}")
        else:
            responses.append("The Airbnb specialist did not return any options.")

        return {
            **state,
            "response_chunks": responses,
            "airbnb_output": airbnb_output,
        }

    def _compose_response(self, state: HostGraphState) -> HostGraphState:
        responses = list(state.get("response_chunks", []))
        policy_notes = state.get("policy_notes", [])
        need_airbnb = state.get("need_airbnb", False)
        final_decision = state.get("final_decision")
        weather_output = state.get("weather_output")

        summary_lines: list[str] = []
        if need_airbnb and final_decision == "deny_airbnb":
            summary_lines.append(
                "Because the forecast includes hazardous conditions, I'm pausing Airbnb "
                "recommendations. Consider alternate dates or destinations."
            )
        elif need_airbnb and state.get("airbnb_output"):
            summary_lines.append(
                "Here are some rental ideas that align with the current forecast."
            )
        elif weather_output:
            summary_lines.append("Let me know if you need help planning activities around this weather outlook.")
        else:
            summary_lines.append(
                "I can coordinate weather checks and Airbnb planning whenever you're ready."
            )

        if policy_notes:
            summary_lines.append("Policy summary:")
            summary_lines.extend(f"- {note}" for note in policy_notes)

        responses.append("\n".join(summary_lines))

        return {**state, "response_chunks": responses}


async def initialize_routing_agent() -> RoutingAgent:
    return await RoutingAgent.create(
        remote_agent_addresses=[
            os.getenv("AIR_AGENT_URL", "http://localhost:10002"),
            os.getenv("WEA_AGENT_URL", "http://localhost:10001"),
        ]
    )
