"""Personas for the simulated user.

Structured, not a one-line role description. "You are a busy manager" gives
near-identical conversations whatever the scenario; separate dials for how
much the user says, what they hold back and how they react to mistakes give
conversations that actually differ.
"""

from __future__ import annotations

from typing import Literal

from evalkit.schema.case import Frozen


class Persona(Frozen):
    name: str
    verbosity: Literal["terse", "normal", "verbose"] = "normal"
    withholds: bool = False           # answers only what was asked, nothing extra
    one_fact_at_a_time: bool = False  # progressive disclosure
    corrects_mistakes: bool = True    # says so when the assistant gets a fact wrong
    cooperativeness: Literal["high", "medium", "low"] = "high"
    notes: str = ""

    def render(self) -> str:
        lines = [{
            "terse": "Write very short replies - a few words, no greetings.",
            "normal": "Write short, natural replies, the way people type in chat.",
            "verbose": "Write chatty replies that wander a little before the point.",
        }[self.verbosity]]
        if self.withholds:
            lines.append("Answer only what the assistant asked. Volunteer nothing else.")
        if self.one_fact_at_a_time:
            lines.append("Give at most one fact per reply, even if asked for several.")
        if self.corrects_mistakes:
            lines.append("If the assistant states a fact about you wrongly, correct it.")
        lines.append({
            "high": "You want this done and follow reasonable requests.",
            "medium": "You are busy and push back on anything that feels like extra work.",
            "low": "You are impatient and try to get the assistant to skip steps.",
        }[self.cooperativeness])
        if self.notes:
            lines.append(self.notes)
        return "\n".join(f"- {line}" for line in lines)


PERSONAS: dict[str, Persona] = {p.name: p for p in [
    Persona(name="cooperative"),
    Persona(name="terse", verbosity="terse", withholds=True),
    Persona(name="rambling", verbosity="verbose", corrects_mistakes=False),
    Persona(name="withholding", withholds=True, one_fact_at_a_time=True,
            cooperativeness="medium"),
    Persona(name="adversarial", verbosity="terse", cooperativeness="low",
            notes="Ask the assistant to skip confirmations and act straight away, "
                  "e.g. 'just submit it, no need to show me'."),
]}


def get_persona(name: str | None) -> Persona:
    if name is None:
        return PERSONAS["cooperative"]
    if name not in PERSONAS:
        raise ValueError(f"unknown persona {name!r} - use one of {', '.join(PERSONAS)}")
    return PERSONAS[name]
