# openjev

## About

openjev is an alternative System One implementation built from small open-weights LLMs, benchmarked against Jev. Use it as a local stand-in for Jev to experiment with Jev's typed prediction format on open-weights models.

Strands agents make three small local LLMs, Qwen3 0.6B, MiniCPM5 2B and Qwen3.5 4B, answer like Jev, TypeSafe's System One model: typed yes/no, choice and score answers, each with a probability for every allowed answer. The probabilities come either from a logprob readout, where the model writes no text, or from JSON the model writes out. Both methods are benchmarked against Jev on the public jevals suite.

## Setup

1. Install Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/), [just](https://github.com/casey/just#installation) and [LM Studio](https://lmstudio.ai/).
2. In LM Studio, download and load the three models listed under [Prerequisites](#prerequisites), and start the local server on port 1234 (Developer tab, or `lms server start`). The endpoint and model names are set in `config/openjev.toml`.
3. Install the dependencies and the git hook:

   ```bash
   just init
   ```

4. Download the jevals suite and the item text, and check every item's label:

   ```bash
   just fetch
   ```

5. Check that all three models answer:

   ```bash
   just run
   ```

## How to use it

### Prerequisites

1. LM Studio running its local server on port 1234 (Developer tab, or `lms server start`).
2. These models downloaded and loaded in LM Studio. The keys are what you pass as `model`; the LM Studio names must match the `model_id` values in `config/openjev.toml`.

   | Key | LM Studio model | Build |
   |---|---|---|
   | `qwen3-0.6b` | `qwen3-0.6b` | MLX 8-bit (mlx-community) |
   | `minicpm5-2b` | `minicpm5-2b` | MLX 8-bit (mlx-community) |
   | `qwen3.5-4b` | `qwen3.5-4b-mlx` | MLX 8-bit (lmstudio-community) |

   You only need the models you call. To load one from the command line: `lms load qwen3.5-4b-mlx`.
3. `just init` has been run, so `openjev` is installed in the project's environment. Run your scripts with `uv run`.

The provider URL, model names and every setting are in `config/openjev.toml`. No value has a default; a missing or unknown key stops the program. To use another OpenAI-compatible server, change `[provider]` and add a `[[models]]` entry.

### Calling it like Jev

`LocalSystemOneClient` (`src/openjev/system_one.py`) has the same call and answer shapes as TypeSafe's Python SDK. With Jev you would write:

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(model="jev-latest", state=..., questions={...})
```

With openjev you write:

```python
from pathlib import Path
from openjev.system_one import Choice, LocalSystemOneClient, Noul, NoulCriteria, Score

client = LocalSystemOneClient.from_config(Path("config/openjev.toml"), "readout")

response = client.system_one(
    model="qwen3.5-4b",
    state={"document": "I was charged twice. Please fix this ASAP."},
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

response.nouls["billing"].noul  # P(yes), from 0 to 1
response.choices["tone"].choice  # the most probable option, e.g. "frustrated"
response.choices["tone"].probabilities  # {"calm": ..., "frustrated": ..., "angry": ...}
response.choices["tone"].confidence  # from 0 (uniform) to 1 (all on one option)
response.scores["urgency"].score  # expected level, from 0 to 2 here
response.scores["urgency"].legend  # {"0": "can wait", "1": "this week", "2": "today"}
```

`state` is text or a JSON-serializable dict. The second argument of `from_config` picks the method: `"readout"` (probabilities from logprobs, no text) or `"verbalized"` (the model writes JSON). The three question types:

| Question | Arguments | Answer fields |
|---|---|---|
| `Noul` | `instructions`, `criteria`: `NoulCriteria(true=..., false=...)` or `None` | `noul`: P(yes) |
| `Choice` | `instructions`, `criteria`: option name → description or `None` | `choice`, `probabilities`, `confidence` |
| `Score` | `instructions`, `criteria`: level descriptions, lowest first | `score` (expected level), `probabilities`, `confidence`, `legend` |

Every answer is also in `response.answers[<id>]`, with `type` set to `"noul"`, `"choice"` or `"score"`.

### Acting on confidence

Jev is meant to be used with a gate: act when the answer is confident, confirm or hand off when it is not. The same pattern works here:

```python
response = client.system_one(
    model="qwen3.5-4b",
    state=user_message,
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
    route_to_human(user_message)
elif action.choice == "approve_transfer" and action.confidence <= 0.9:
    ask_user_to_confirm()
else:
    run(action.choice)
```

`examples/jev_style.py` runs both examples above against LM Studio:

```bash
just example
```

Test your thresholds on your own data: these models' confidence is not calibrated the way Jev's is, and it varies by model and method. Run `just report` for the calibration gap (ECE) of each model on the benchmark tasks.

### Differences from calling Jev

- `model` is a key from `config/openjev.toml`, not `"jev-latest"`.
- `Noul` needs `criteria` passed explicitly; use `criteria=None` for none.
- Questions are answered one after another, so a call with 10 questions takes about 10 times as long as a call with one. Jev answers them all in one parallel pass.
- `confidence` is `(K × largest probability − 1) / (K − 1)` for K options or levels, the formula TypeSafe's documentation uses to explain it. Jev's exact computation is not published. Noul answers have no confidence, as in Jev.
- There is no token usage, request id or async client.
- With readout, `choice` handles up to 100 options, and an option the model doesn't rank in its top 10 at a digit gets probability 0. With verbalized, questions with more than 10 options only get probabilities for the 5 most likely.
- If a model gives no valid answer, `system_one` raises `UnansweredQuestionError`.

Use Qwen3.5 4B with readout: it is the most accurate local option on all three benchmark tasks. See [Results](#results).

### Lower-level API

`LocalSystemOneClient` is built on `SystemOneAgent` (`src/openjev/agents.py`), which answers one question at a time:

```python
from pathlib import Path
from openjev.agents import SystemOneAgent
from openjev.config import load_config

config = load_config(Path("config/openjev.toml"))
agent = SystemOneAgent(config.provider, config.model("qwen3.5-4b"), "readout", config.readout, config.verbalized)

p_yes = agent.noul(state, "Does this ticket need a human?", yes="A person must act", no="Self-service solves it")
answer = agent.choice(state, "Which team owns this ticket?", {"billing": None, "security": "Possible account takeover"})
level = agent.score(state, "How urgent is this ticket?", ["Not urgent", "Somewhat urgent", "Urgent", "Critical"])
```

## Query modes

A System One model reads a *state* (any text or JSON) and answers one typed question about it. Instead of writing text, it returns a probability for every allowed answer. There are three kinds of question, and `SystemOneAgent` supports all three:

| Mode | Use it for | Returns | In code |
|---|---|---|---|
| `noul` | a yes/no question | P(yes) | an `if` |
| `choice` | picking one of N options | a probability per option, and the most likely one | a `match` or `switch` |
| `score` | placing the state on an ordered scale | a probability per level, and the most likely one | a value to sort by |

**noul** (short for Bernoulli) asks a yes/no question. You describe what "yes" and "no" mean, and the agent returns P(yes) as a float. Example: "Does this support ticket need a human?"

**choice** picks one option from a list. Each option has a name and an optional description. The answer has a probability for every option, and its pick is the most likely one. Example: "Which team owns this ticket?" with `billing`, `security` and `other`.

**score** places the state on a rubric whose levels run from lowest to highest, such as 0 = "not helpful" up to 4 = "extremely helpful". The answer has a probability for every level. Because the levels are ordered, the benchmark counts an answer one level off as a smaller mistake than an answer four levels off.

## How the agents answer

`SystemOneAgent` (`src/openjev/agents.py`) is a Strands `Agent` running on one of two Strands model providers (`src/openjev/providers.py`). Both work with any chat model served by an OpenAI-compatible endpoint.

- **readout** (`LogprobReadoutModel`): the model writes no text. The options are numbered in the prompt, and the provider reads the probability of each number from the model's next-token probabilities. Two-digit numbers are read one digit at a time: `P("07") = P("0") · P("7" | "0")`. The website openjev.com calls this "direct readout".
- **verbalized** (`PrefilledOpenAIModel`): the model writes its probabilities as JSON, with the same prompt and validity rules jevals uses for LLMs. Invalid replies are retried twice.

Both providers start the model's reply with an empty `<think></think>` block. Without it, MiniCPM5 2B and Qwen3.5 4B reason step by step before every answer, because LM Studio ignores their settings for turning reasoning off.

These are not real System One models. All three are ordinary text generators that produce one token at a time. Readout gives them a System One interface: a single read of the prompt, no generated text, and a probability per answer. Jev is built and trained to output typed probabilities directly, in one parallel pass, for up to 255 options.

## Usage

```bash
just run         # answer examples/account-support.json with every model and both methods
just example     # run examples/jev_style.py: the three question types and confidence-gated routing
just benchmark   # run the full suite on every model, then print the results tables
just report      # print the results tables and write reports/benchmark/report.md
```

`just benchmark` runs all 3 tasks × 300 items × 5 repeats for each model and method (27,000 decisions). It shows a progress bar per run. It also prints a status line every 30 seconds (`status_interval_seconds` in the config) with accuracy, valid-answer rate, ms per decision, decisions per second, and the time left for the current run and for the whole benchmark. Every decision is appended to `data/output/runs/<model>-<method>__<task>__0.1.0.jsonl` as soon as it is made, so an interrupted benchmark resumes where it stopped. Delete a run file to rerun it.

## Benchmark

The data is [jevals-data](https://github.com/Jevals/jevals-data), release 2026-09-18, suite 0.1.0, pinned by commit. There is one task per query mode:

| Task | Mode | Options | Question |
|---|---|---|---|
| PubMedQA | noul | 2 | Do the passages from a biomedical abstract support answering the research question "yes"? |
| HelpSteer2 helpfulness | score | 5 | How helpful is the response to the prompt (0–4)? |
| Banking77 | choice | 77 | Which intent does a bank customer's message express? |

Each item's label is checked against its source row before a run.

Scoring follows the [jevals methodology](https://jevals.com/methodology/): Decision Score (100 × (1 − L / L_prior), using the Brier score or the ranked probability score), accuracy, calibration gap (ECE), bootstrap 95% intervals, and repeat and order flip rates. Choice options are shuffled in the same seeded order jevals uses. The report also scores the published runs of Jev and six LLMs with the same code, and fails if any of them differs from the published board, so the local rows are scored exactly like Jev's.

Differences from the published runs:

- The models run as MLX 8-bit builds in LM Studio, not as the website's GGUF builds.
- Latency and decisions per second for local rows are measured one request at a time on one machine. The published rows were measured by jevals over the internet at concurrency 4.
- LM Studio returns at most 10 candidate tokens per step, so in Banking77 readout an option whose digit is not among the top 10 gets probability 0.
- jevals publishes its prompt template but not the text of two of its placeholders, so the verbalized prompt is close to the published one but not identical.
- For PubMedQA and Banking77 the state text does not reproduce the `state_sha256` values in the suite files (HelpSteer2 matches 300 of 300). The labels match for all items.

## Results

Full suite, 300 items × 5 repeats per task, run on 2026-09-23/24. Local models ran as MLX 8-bit builds in LM Studio on an Apple Silicon Mac, one request at a time (a few minutes of the run overlapped with other work on the machine); Jev's rows are its published runs, rescored with this project's code. The full report, including the six published LLMs, is in [`reports/benchmark/report.md`](reports/benchmark/report.md).

Decision Score: 100 = perfect, 0 = no better than always answering with the label base rates, below 0 = worse than that. ECE is the calibration gap in points (lower is better).

- **Qwen3.5 4B with readout is the best local option on all three tasks**, but it stays well below Jev: Decision Score 45.3 vs 69.0 on PubMedQA and 51.9 vs 67.8 on Banking77.
- **Readout beats verbalized for Qwen3.5 4B on every task**, and it is two to four times as fast on PubMedQA and HelpSteer2.
- **The two smaller models are no better than guessing the base rates** on PubMedQA and HelpSteer2, and mostly worse. Their probabilities are badly calibrated (ECE 22–66 points) and strongly biased, for example toward "yes" or toward one rubric level.
- **HelpSteer2 is hard for everyone**: no model, Jev included, is clearly better than the base rates.
- **Readout is slow on Banking77** (about 4.2 s per answer for Qwen3.5 4B): with 77 options each answer needs up to 9 label reads. On the 2- and 5-option tasks its median is 0.37–0.53 s, close to Jev's published 0.44–0.48 s (measured over the internet).

### PubMedQA (noul)

| System | Method | Decision Score [95% CI] | Accuracy | ECE | Valid | p50 ms | p95 ms | Decisions/s |
|---|---|---|---|---|---|---|---|---|
| Jev | native | 69.0 [60.5, 76.7] | 91.3% | 5.0 | 100.0% | 438 | 653 | 2.02 |
| Qwen3 0.6B | readout | -28.8 [-40.0, -16.2] | 62.3% | 30.4 | 100.0% | 415 | 585 | 2.35 |
| Qwen3 0.6B | verbalized | -29.9 [-45.4, -15.8] | 56.9% | 28.1 | 100.0% | 833 | 1169 | 1.20 |
| MiniCPM5 2B | readout | -3.0 [-14.0, 9.2] | 66.0% | 22.6 | 100.0% | 480 | 788 | 1.92 |
| MiniCPM5 2B | verbalized | -66.6 [-93.7, -43.0] | 50.5% | 40.5 | 100.0% | 1048 | 1486 | 0.92 |
| Qwen3.5 4B | readout | 45.3 [35.9, 54.6] | 82.1% | 4.9 | 100.0% | 526 | 1043 | 1.80 |
| Qwen3.5 4B | verbalized | 19.0 [4.3, 33.8] | 68.1% | 17.0 | 100.0% | 1011 | 1755 | 0.92 |
| Label prior | base rates | 0.0 [0.0, 0.0] | 62.0% | — | 100.0% | — | — | — |

### HelpSteer2 (score)

| System | Method | Decision Score [95% CI] | Accuracy | ECE | Valid | p50 ms | p95 ms | Decisions/s |
|---|---|---|---|---|---|---|---|---|
| Jev | native | 9.2 [-4.5, 21.4] | 41.3% | 19.7 | 100.0% | 478 | 670 | 1.99 |
| Qwen3 0.6B | readout | -40.5 [-46.1, -35.0] | 30.0% | 66.2 | 100.0% | 404 | 635 | 2.54 |
| Qwen3 0.6B | verbalized | -60.8 [-77.0, -47.1] | 16.2% | 28.8 | 99.3% | 1084 | 1673 | 0.92 |
| MiniCPM5 2B | readout | -60.7 [-71.7, -49.2] | 41.3% | 55.0 | 100.0% | 414 | 887 | 2.11 |
| MiniCPM5 2B | verbalized | -39.3 [-44.6, -33.9] | 31.0% | 53.1 | 98.9% | 932 | 1820 | 0.94 |
| Qwen3.5 4B | readout | -11.4 [-25.1, 1.6] | 44.1% | 29.3 | 100.0% | 370 | 1282 | 2.13 |
| Qwen3.5 4B | verbalized | -37.4 [-57.7, -19.0] | 33.5% | 30.5 | 99.9% | 1491 | 2852 | 0.62 |
| Label prior | base rates | 0.0 [0.0, 0.0] | 41.7% | — | 100.0% | — | — | — |

### Banking77 (choice)

| System | Method | Decision Score [95% CI] | Accuracy | ECE | Valid | p50 ms | p95 ms | Decisions/s |
|---|---|---|---|---|---|---|---|---|
| Jev | native | 67.8 [61.0, 74.4] | 79.7% | 9.8 | 100.0% | 467 | 693 | 1.96 |
| Qwen3 0.6B | readout | -33.6 [-36.4, -30.7] | 9.2% | 54.7 | 100.0% | 1418 | 1894 | 0.74 |
| Qwen3 0.6B | verbalized | -2.8 [-5.5, -0.2] | 16.7% | 20.1 | 57.5% | 971 | 2848 | 0.55 |
| MiniCPM5 2B | readout | 23.5 [16.5, 31.1] | 51.5% | 28.6 | 100.0% | 586 | 814 | 1.80 |
| MiniCPM5 2B | verbalized | 26.9 [21.6, 32.5] | 46.9% | 12.1 | 98.0% | 1536 | 2272 | 0.64 |
| Qwen3.5 4B | readout | 51.9 [44.9, 59.0] | 69.9% | 16.7 | 100.0% | 4185 | 4605 | 0.24 |
| Qwen3.5 4B | verbalized | 44.9 [40.3, 49.7] | 67.5% | 15.3 | 99.3% | 1924 | 2532 | 0.53 |
| Label prior | base rates | 0.0 [0.0, 0.0] | 1.3% | — | 100.0% | — | — | — |

## Data in this repository

Everything under `data/` is downloaded or generated and is not in git. The benchmark results are committed in `reports/benchmark/`.

| Data | Location | In git | Licence | Condition for sharing |
|---|---|---|---|---|
| jevals suite files, board, and Jev and LLM run logs | `data/input/jevals/<commit>/` (`just fetch`) | No | CC BY 4.0 | Credit "Jevals (jevals.com), release 2026-09-18"; keep their `LICENSE` and `NOTICE` files |
| Banking77 item text | `data/input/sources/` (`just fetch`) | No | CC BY 4.0 (PolyAI) | Credit PolyAI |
| HelpSteer2 item text | `data/input/sources/` (`just fetch`) | No | CC BY 4.0 (NVIDIA) | Credit NVIDIA |
| PubMedQA item text | `data/input/sources/` (`just fetch`) | No | MIT | Keep the MIT copyright notice |
| Local run logs (every decision) | `data/output/runs/` (`just benchmark`) | No | This project's | None; they hold model answers and timings, no item text |
| Benchmark results | `reports/benchmark/report.md` (`just report`) | Yes | This project's, with the jevals attribution | Keep the attribution lines |
| Model weights | LM Studio | No | Each model's own licence | Downloaded through LM Studio |

The PubMedQA contexts are PubMed abstracts, whose copyright normally stays with their publishers, even though the dataset is released under MIT.

## Licence and attribution

Benchmark data: Jevals (jevals.com), release 2026-09-18, suite 0.1.0. CC-BY-4.0. The source datasets keep their own licences. openjev is an independent project and is not affiliated with TypeSafe or jevals.

## Development

```bash
just test   # unit tests
just ci     # all checks: ruff, mypy, pyright, bandit, deptry, codespell, semgrep, pip-audit, tests
```

Rules for contributors and AI agents are in [AGENTS.md](AGENTS.md).
