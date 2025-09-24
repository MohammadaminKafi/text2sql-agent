"""
Dynamic User Interaction Interface

This module provides an abstract interface for user interactions that can work
both in console mode (for direct usage) and in web mode (via backend callback).
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Callable, Any
import asyncio
from dataclasses import dataclass


@dataclass
class InteractionResponse:
    """Response from user interaction"""
    answer: str
    is_preset: bool = False
    preset_index: Optional[int] = None


class UserInteractionInterface(ABC):
    """Abstract interface for user interactions"""
    
    @abstractmethod
    def ask_user(self, question: str, preset_answers: Optional[List[str]] = None) -> str:
        """Ask user a question and return their response"""
        pass


class ConsoleInteraction(UserInteractionInterface):
    """Console-based interaction (original behavior)"""
    
    def __init__(self, preset_answers: Optional[List[str]] = None):
        self.default_presets = preset_answers or [
            "Yes",
            "No", 
            "I don't know",
            "Doesn't matter",
            "Maybe",
            "Absolutely!",
            "Absolutely not",
        ]
    
    def ask_user(self, question: str, preset_answers: Optional[List[str]] = None) -> str:
        """Console implementation using input()"""
        presets = preset_answers or self.default_presets
        
        # Display the question
        print("\n" + "─" * 60)
        print(f"Agent asked for more clarification ➜ {question}\n")

        # Show the preset menu
        for idx, ans in enumerate(presets, start=1):
            print(f"[{idx}] {ans}")
        print("[0] Other…")

        # Keep asking until we get a valid response
        while True:
            choice = input("\nChoose a number or press Enter for 'Other': ").strip()

            if choice == "" or choice == "0":
                # Free-form path
                custom = input("Your custom answer: ").strip()
                if custom:
                    return custom
                print("⚠️  Empty input—please type something.")
                continue

            # Numeric choice → preset answer
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(presets):
                    return presets[idx - 1]

            # Anything else is invalid ⇒ loop again
            print("Invalid selection. Try again.")


class WebInteraction(UserInteractionInterface):
    """Web-based interaction via callback to backend"""
    
    def __init__(self, callback_function: Callable[[str, List[str]], str]):
        """
        Initialize with a callback function that handles the web interaction
        
        Args:
            callback_function: Function that takes (question, preset_answers) 
                             and returns user's answer (can be sync or async)
        """
        self.callback = callback_function
        self.default_presets = [
            "Yes",
            "No", 
            "I don't know",
            "Doesn't matter",
            "Maybe", 
            "Absolutely!",
            "Absolutely not",
        ]
    
    def ask_user(self, question: str, preset_answers: Optional[List[str]] = None) -> str:
        """Web implementation via callback (handles both sync and async)"""
        presets = preset_answers or self.default_presets
        
        # Check if callback is async
        if asyncio.iscoroutinefunction(self.callback):
            # Handle async callback in sync context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're in an event loop, use run_until_complete via thread
                    import concurrent.futures
                    import threading
                    
                    def run_in_new_loop():
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            return new_loop.run_until_complete(self.callback(question, presets))
                        finally:
                            new_loop.close()
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_new_loop)
                        return future.result()
                else:
                    return loop.run_until_complete(self.callback(question, presets))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(self.callback(question, presets))
        else:
            # Sync callback
            return self.callback(question, presets)


class AsyncWebInteraction(UserInteractionInterface):
    """Async web-based interaction for async contexts"""
    
    def __init__(self, async_callback: Callable[[str, List[str]], Any]):
        """
        Initialize with an async callback function
        
        Args:
            async_callback: Async function that takes (question, preset_answers)
                           and returns user's answer (can be coroutine or future)
        """
        self.callback = async_callback
        self.default_presets = [
            "Yes",
            "No",
            "I don't know", 
            "Doesn't matter",
            "Maybe",
            "Absolutely!",
            "Absolutely not",
        ]
    
    def ask_user(self, question: str, preset_answers: Optional[List[str]] = None) -> str:
        """Async web implementation"""
        presets = preset_answers or self.default_presets
        
        # Handle async callback in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, but ask_user needs to be sync
                # This is a limitation - the caller should use the async version
                raise RuntimeError("Cannot call async callback from sync context")
            else:
                return loop.run_until_complete(self.callback(question, presets))
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self.callback(question, presets))


# Global interaction interface instance
_interaction_interface: UserInteractionInterface = ConsoleInteraction()


def set_interaction_interface(interface: UserInteractionInterface) -> None:
    """Set the global interaction interface"""
    global _interaction_interface
    _interaction_interface = interface


def get_interaction_interface() -> UserInteractionInterface:
    """Get the current interaction interface"""
    return _interaction_interface


def ask_user(question: str, preset_answers: Optional[List[str]] = None) -> str:
    """
    Ask user a question using the current interaction interface
    
    This function replaces the original ask_user function and can work
    in both console and web modes depending on the configured interface.
    """
    return _interaction_interface.ask_user(question, preset_answers)