"""
User clarification utilities with support for both console and web interactions.
"""

from .interaction_interface import ask_user as _ask_user_interface

# Backward compatibility - keep the same function signature
def ask_user(question: str) -> str:
    """
    Print `question`, then show a numbered list of preset replies plus an
    "Other…" option. Return the chosen answer (or free-form input).

    Now uses the interaction interface which can work in console or web mode.
    """
    return _ask_user_interface(question)