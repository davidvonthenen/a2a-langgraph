"""Routing agent backed by the OpenAI SDK."""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

import httpx
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    Part,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TextPart,
    DataPart,
    FilePart,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .remote_agent_connection import RemoteAgentConnections

load_dotenv()

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

# Remote card discovery may involve slow remote agents. Give the HTTP client a
# generous timeout so initialization succeeds in those cases.
DEFAULT_HTTP_TIMEOUT = httpx.Timeout(120.0, connect=30.0)


class RoutingAgent:
    """Delegates user requests to remote agents using OpenAI for planning."""

    SYSTEM_PROMPT = (
        "You are a routing assistant coordinating specialized agents. "
        "Decide whether to answer the user directly or delegate to one of the remote agents. "
        "Respond with JSON only."
    )

    def __init__(
        self,
        *,
        client: AsyncOpenAI | None = None,
        model: str | None = None,
    ) -> None:
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self._client = client or AsyncOpenAI()
        self._model = model or os.getenv(
            "OPENAI_ROUTER_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-nano")
        )
        self._session_history: dict[str, list[ChatCompletionMessageParam]] = {}
        self._session_context_ids: dict[tuple[str, str], str] = {}
        logger.info("RoutingAgent initialized with model %s", self._model)

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
        plan = await self._plan_action(message, session_id)
        responses: list[str] = []

        action = plan.get("action")
        if action == "delegate":
            agent_name = plan.get("agent")
            task_text = plan.get("task")
            if not agent_name or not task_text:
                fallback = (
                    "I could not determine which specialist to use. "
                    "Could you rephrase your request?"
                )
                self._append_to_history(session_id, fallback)
                return [fallback]

            responses.append(f"Delegating to {agent_name}...")
            task = await self._send_message(agent_name, task_text, session_id)
            agent_output = self._extract_task_output(task)
            final_message = await self._summarize_response(
                message, agent_name, agent_output, session_id
            )
            responses.append(final_message)
            return responses

        if action == "ask_user":
            question = plan.get("question") or "Could you share more details?"
            self._append_to_history(session_id, question)
            responses.append(question)
            return responses

        reply = plan.get("message") or "I'm not sure how to help with that."
        self._append_to_history(session_id, reply)
        responses.append(reply)
        return responses

    async def _plan_action(self, message: str, session_id: str) -> dict[str, Any]:
        agents_description = "\n".join(
            f"- {card.name}: {card.description or 'No description provided.'}"
            for card in self.cards.values()
        ) or "- No remote agents available"
        system_prompt = (
            f"{self.SYSTEM_PROMPT}\n\nAvailable agents:\n{agents_description}\n\n"
            "Respond with one of the following JSON structures:\n"
            '{"action": "delegate", "agent": "Agent Name", "task": "Detailed task"}\n'
            '{"action": "ask_user", "question": "Clarifying question"}\n'
            '{"action": "respond", "message": "Assistant reply"}'
        )

        history = self._history_for_session(session_id)
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": message},
        ]
        history.append({"role": "user", "content": message})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            # temperature=0.2, # BAD
        )
        raw_content = response.choices[0].message.content or "{}"
        try:
            plan = json.loads(raw_content)
        except json.JSONDecodeError:
            logger.error("Router response was not valid JSON: %s", raw_content)
            plan = {"action": "respond", "message": raw_content}

        return plan

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

    async def _summarize_response(
        self,
        user_message: str,
        agent_name: str,
        agent_output: str,
        session_id: str,
    ) -> str:
        summary_prompt = (
            "You are the host assistant. Summarize the remote agent's reply for the user. "
            "If the remote agent output is empty, politely inform the user that the specialist "
            "did not return any information."
        )
        messages = [
            {"role": "system", "content": summary_prompt},
            {
                "role": "user",
                "content": (
                    f"Original user request:\n{user_message}\n\n"
                    f"Remote agent ({agent_name}) responded:\n{agent_output or 'No response'}"
                ),
            },
        ]
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            # temperature=0.2, # BAD
        )
        content = (
            response.choices[0].message.content
            or "The remote agent did not provide any additional details."
        )
        self._append_to_history(session_id, content)
        return content

    def _history_for_session(
        self, session_id: str
    ) -> list[ChatCompletionMessageParam]:
        return self._session_history.setdefault(session_id, [])

    def _append_to_history(self, session_id: str, content: str) -> None:
        history = self._history_for_session(session_id)
        history.append({"role": "assistant", "content": content})

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


async def initialize_routing_agent() -> RoutingAgent:
    return await RoutingAgent.create(
        remote_agent_addresses=[
            os.getenv("AIR_AGENT_URL", "http://localhost:10002"),
            os.getenv("WEA_AGENT_URL", "http://localhost:10001"),
        ]
    )
