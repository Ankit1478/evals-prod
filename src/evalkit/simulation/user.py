"""The simulated user: answers the agent in multi-turn cases.

Two modes. A SCRIPT replays fixed replies in order - free, and identical on
every trial, so a failure is the agent's and never the user's. Without a
script a model plays the user from the case's goal, facts and persona.

The model only ever sees the conversation TEXT - never tool calls or tool
results. A real user does not see those either, and letting the simulator
read them leaks answers the user could not know.
"""

from __future__ import annotations

import hashlib

from evalkit.providers.base import Provider
from evalkit.schema.case import UserSim
from evalkit.simulation.personas import Persona

DONE = "[DONE]"

PROMPT = """You are role-playing the USER in a chat with an AI assistant. You
are not the assistant, and you never help it with its job.

Your goal: {goal}

What you know (share it when asked, in your own words; do not invent other facts):
{facts}

How you write:
{persona}

Reply with only the next message the user would type - no quotes, no labels.
When your goal is done, or the assistant is not waiting on anything from you,
reply with exactly {done}"""


def render_transcript(messages: list[dict]) -> str:
    """Only what a person in the chat would see: the typed messages."""
    lines = []
    for m in messages:
        if m["role"] in ("user", "assistant") and (m.get("content") or "").strip():
            who = "USER (you)" if m["role"] == "user" else "ASSISTANT"
            lines.append(f"{who}: {m['content'].strip()}")
    return "\n\n".join(lines)


class SimulatedUser:
    version = "1"

    def __init__(self, sim: UserSim, persona: Persona,
                 provider: Provider | None = None) -> None:
        if not sim.script and provider is None:
            raise ValueError("this case has no user_sim.script, so a model must play "
                             "the user - pass --user-model")
        self.sim = sim
        self.persona = persona
        self.provider = provider
        self._next = 0
        facts = "\n".join(f"- {k}: {v}" for k, v in sim.facts.items()) or "- nothing beyond your goal"
        self.system = PROMPT.format(goal=sim.goal, facts=facts,
                                    persona=persona.render(), done=DONE)

    async def reply(self, messages: list[dict]) -> str | None:
        """The user's next message, or None when the user is finished."""
        if self.sim.script:
            if self._next >= len(self.sim.script):
                return None
            text = self.sim.script[self._next]
            self._next += 1
            return None if text.strip() == DONE else text

        prompt = ("Conversation so far:\n\n" + render_transcript(messages)
                  + "\n\nWrite the user's next message.")
        out = await self.provider.complete([{"role": "user", "content": prompt}],
                                           system=self.system, max_tokens=2000)
        text = out.text.strip()
        if not text or DONE in text:
            return None
        return text

    def describe(self) -> dict:
        """For the manifest: the simulator is part of the measurement, so its
        exact prompt and model are recorded."""
        return {
            "version": self.version,
            "persona": self.persona.name,
            "mode": "script" if self.sim.script else "model",
            "model": self.provider.describe() if self.provider and not self.sim.script else None,
            "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest()[:12],
        }
