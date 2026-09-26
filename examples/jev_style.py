"""Call local models the way you would call Jev with the TypeSafe Python SDK.

Run with: uv run examples/jev_style.py --config config/llm-system-one.toml --model qwen3.5-4b --method readout
"""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_system_one.system_one import Choice, LocalSystemOneClient, Noul, NoulCriteria, Score


def triage(client: LocalSystemOneClient, model: str, ticket: str) -> None:
    """Ask all three question types about one support ticket, as in the TypeSafe SDK quickstart."""
    response = client.system_one(
        model=model,
        state={"document": ticket},
        questions={
            "billing": Noul(instructions="Is this ticket about billing?", criteria=None),
            "repeat": Noul(
                instructions="Has the customer contacted support about this before?",
                criteria=NoulCriteria(true="Mentions a prior attempt or ticket", false="No sign of previous contact"),
            ),
            "tone": Choice(
                instructions="What is the customer's tone?",
                criteria={"calm": None, "frustrated": None, "angry": None},
            ),
            "urgency": Score(
                instructions="How urgent is this ticket?",
                criteria=["can wait", "this week", "today"],
            ),
        },
    )
    print(f"ticket:  {ticket}")
    print(f"billing: P(yes) = {response.nouls['billing'].noul:.2f}")
    print(f"repeat:  P(yes) = {response.nouls['repeat'].noul:.2f}")
    tone = response.choices["tone"]
    print(f"tone:    {tone.choice} (confidence {tone.confidence:.2f}) {tone.probabilities}")
    urgency = response.scores["urgency"]
    print(f"urgency: {urgency.score:.2f} on 0-2 (confidence {urgency.confidence:.2f}) {urgency.legend}")


def route(client: LocalSystemOneClient, model: str, message: str) -> str:
    """Confidence-gated routing: act when the model is sure, confirm or hand off when it is not."""
    response = client.system_one(
        model=model,
        state=message,
        questions={
            "action": Choice(
                instructions="What is the user trying to do?",
                criteria={
                    "check_balance": "View account balance",
                    "approve_transfer": "Approve the pending withdrawal request",
                    "support": "Get help with an issue",
                },
            ),
        },
    )
    action = response.choices["action"]
    if action.confidence < 0.5:
        return f"route to a human (unsure: {action.choice}, confidence {action.confidence:.2f})"
    if action.choice == "approve_transfer" and action.confidence <= 0.9:
        return f"ask the user to confirm the transfer (confidence {action.confidence:.2f})"
    return f"do {action.choice} (confidence {action.confidence:.2f})"


def main() -> None:
    """Run the triage and routing examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model key from the config, e.g. qwen3.5-4b")
    parser.add_argument("--method", choices=["readout", "verbalized"], required=True)
    args = parser.parse_args()
    client = LocalSystemOneClient.from_config(args.config, args.method)

    triage(client, args.model, "I was charged twice. Please fix this ASAP. This is my third email about it.")
    print()
    for message in ["How much money is in my account?", "Yes, go ahead and send the 500 euros", "hmm"]:
        print(f"{message!r:45} -> {route(client, args.model, message)}")


if __name__ == "__main__":
    main()
