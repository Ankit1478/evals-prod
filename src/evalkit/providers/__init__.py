"""Model access. One spec string picks the vendor and the model:

    anthropic:claude-opus-5               Claude API
    bedrock:anthropic.claude-sonnet-5     Claude on Amazon Bedrock
    foundry:claude-opus-5                 Claude on Microsoft Foundry (Azure)
    openai:gpt-4o-mini                    OpenAI
    azure:<deployment>                    Azure OpenAI

Options go after a `?`:  anthropic:claude-opus-5?effort=low
A bare model name still works: `claude-...` means the Claude API, anything
else means OpenAI - so `--model gpt-4o-mini` keeps meaning what it meant.
"""

from __future__ import annotations

from urllib.parse import parse_qsl

from evalkit.providers.anthropic import AnthropicProvider
from evalkit.providers.base import (Completion, Provider, ProviderError,
                                    ProviderToolCall, add_usage)
from evalkit.providers.openai import OpenAIProvider

PLATFORMS = ("anthropic", "bedrock", "foundry", "openai", "azure")


def parse_spec(spec: str) -> tuple[str, str | None, dict[str, str]]:
    """'bedrock:anthropic.claude-opus-5?effort=low' -> ('bedrock', 'anthropic.claude-opus-5', {'effort': 'low'})"""
    spec, _, query = spec.strip().partition("?")
    options = dict(parse_qsl(query, strict_parsing=bool(query)))
    platform, sep, model = spec.partition(":")
    if not sep:
        # Bare model name. Note "anthropic.claude-..." is a Bedrock id, which
        # needs the bedrock: prefix - there is no way to guess the region.
        platform, model = ("anthropic" if spec.startswith("claude-") else "openai"), spec
    if platform not in PLATFORMS:
        raise ProviderError(f"unknown provider {platform!r} - use one of {', '.join(PLATFORMS)}")
    return platform, model or None, options


def get_provider(spec: str) -> Provider:
    platform, model, options = parse_spec(spec)
    if platform in ("anthropic", "bedrock", "foundry"):
        unknown = set(options) - {"effort"}
        if unknown:
            raise ProviderError(f"unknown option(s) for {platform}: {', '.join(sorted(unknown))}")
        return AnthropicProvider(platform, model, effort=options.get("effort"))
    if options:
        raise ProviderError(f"{platform} takes no options, got: {', '.join(sorted(options))}")
    return OpenAIProvider(platform, model)


__all__ = ["AnthropicProvider", "Completion", "OpenAIProvider", "PLATFORMS",
           "Provider", "ProviderError", "ProviderToolCall", "add_usage",
           "get_provider", "parse_spec"]
