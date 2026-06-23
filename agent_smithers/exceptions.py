"""Custom exception types raised across the application.

Defines the small set of domain-specific exceptions used to signal
configuration and provider problems, keeping error handling distinct from
generic built-in exceptions.
"""
class ConfigError(Exception):
    """Configuration is invalid or missing required fields.

    Raised while loading or validating settings — for example a missing env
    file, malformed JSON, or an absent required value — so callers can fail
    fast with an actionable message.
    """


class ProviderError(Exception):
    """An LLM provider is misconfigured or unusable.

    Raised when a provider cannot be resolved or lacks the credentials/base
    URL it needs, distinguishing provider setup faults from general errors.
    """

