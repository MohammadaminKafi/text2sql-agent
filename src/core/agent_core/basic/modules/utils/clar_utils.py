_PRESET_ANSWERS = [
    "Yes",
    "No",
    "I don't know",
    "Doesn't matter",
    "Maybe",
    "Absolutely!",
    "Absolutely not",
]


def ask_user(question: str) -> str:
    """
    Print `question`, then show a numbered list of preset replies plus an
    "Other…" option. Return the chosen answer (or free-form input).

    Works in any CLI / notebook environment that supports `input()`.
    """
    # Display the question
    print("\n" + "─" * 60)
    print(f"Agent asked for more clarification ➜ {question}\n")

    # Show the preset menu
    for idx, ans in enumerate(_PRESET_ANSWERS, start=1):
        print(f"[{idx}] {ans}")
    print("[0] Other…")  # sentinel for custom input

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
            if 1 <= idx <= len(_PRESET_ANSWERS):
                return _PRESET_ANSWERS[idx - 1]

        # Anything else is invalid ⇒ loop again
        print("Invalid selection. Try again.")