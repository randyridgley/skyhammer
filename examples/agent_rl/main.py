"""A number-guessing agent for multi-turn RL (examples/43_agent_rl.py trains against it).

Each task row holds a secret number between 1 and 100. The model being trained plays
through one tool, ``guess(n)``, which answers "higher", "lower" or "correct", for up to
seven guesses. The reward is outcome first: any solve beats any loss, and a faster solve
beats a slower one.

Deploy it to Amazon Bedrock AgentCore (with this folder's requirements.txt) and give its
runtime role the AWS managed policy AmazonSageMakerJobRuntimeAccess, which lets it call
the model being trained and report rewards.
"""

import json

from bedrock_agentcore import BedrockAgentCoreApp

from skyhammer.agent_rl import Rollout, rollout_handler

MAX_GUESSES = 7
SYSTEM = (
    "You are playing a guessing game. A secret whole number between 1 and 100 is hidden. "
    "Call the guess tool with one number at a time; it tells you whether the secret is "
    "higher, lower, or correct. Find it in as few guesses as you can."
)
GUESS_TOOL = {
    "type": "function",
    "function": {
        "name": "guess",
        "description": "Guess the secret number.",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer", "description": "Your guess, 1-100"}},
            "required": ["n"],
        },
    },
}

app = BedrockAgentCoreApp()


def reward(solved_on: int | None, valid_guesses: int) -> float:
    if solved_on is not None:
        return 1.0 + 0.5 * (MAX_GUESSES - solved_on) / (MAX_GUESSES - 1)  # 1.5 .. 1.0
    return -0.5 if valid_guesses else -1.0  # lost, or never played


@app.entrypoint  # outermost: AgentCore's HTTP server calls this
@rollout_handler  # reports the returned reward to the training job
def play(rollout: Rollout) -> float:
    secret = int(rollout.task["secret"])
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Start guessing."},
    ]
    valid = 0
    for _ in range(MAX_GUESSES + 3):  # a few extra turns for malformed calls
        reply = rollout.chat(messages=messages, tools=[GUESS_TOOL])
        msg = reply.choices[0].message
        if not msg.tool_calls:
            messages += [
                {"role": "assistant", "content": msg.content or ""},
                {"role": "user", "content": "Use the guess tool."},
            ]
            continue
        call = msg.tool_calls[0]
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            }
        )
        try:
            n = int(json.loads(call.function.arguments)["n"])
        except (ValueError, KeyError, TypeError):
            answer = "Invalid call: pass an integer n."
        else:
            valid += 1
            if n == secret:
                return reward(valid, valid)
            answer = "higher" if secret > n else "lower"
            if valid == MAX_GUESSES:
                break
        messages.append({"role": "tool", "tool_call_id": call.id, "content": answer})
    return reward(None, valid)


if __name__ == "__main__":
    app.run()
