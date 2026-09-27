# llm-system-one benchmark: local System One agents vs Jev

Benchmark data: Jevals (jevals.com), release 2026-09-18, suite 0.1.0. CC-BY-4.0. Item text: Banking77 (CC BY 4.0, PolyAI), HelpSteer2 (CC BY 4.0, NVIDIA), PubMedQA (MIT).

## What we ran and where the results came from

**We did not call Jev or run the six published reference LLMs ourselves.** Their rows use answer logs published by Jevals, rescored with our code. Only the rows marked **(local)** come from models we ran ourselves through LM Studio: Qwen3 0.6B, MiniCPM5 2B, Qwen3.5 4B and GLM 4.7 Flash, each with readout and verbalized probabilities. The Label prior row is a calculated baseline, not a model run.

`just fetch` downloads the required files from `Jevals/jevals-data`, pinned to commit `21bb47b72814cf661539b313844d2d2e26166e54`, into `data/input/jevals/21bb47b72814cf661539b313844d2d2e26166e54/`. These include Jev's answer logs (`runs/jev__<task>__0.1.0.jsonl`), the other reference logs, and the published board (`releases/2026-09-18/board.json`). The logs contain the published probabilities and timings for each item and repeat.

## How we scored and checked the results

`read_run` and `metrics` in `src/llm_system_one/scoring.py` score both the published logs and our local runs using the Jevals formulas. `check_against_board` checks the recomputed reference Decision Score, accuracy, loss, ECE, repeat/order flip rates, and p50/p95 latency against the published board. Report generation stops if a checked value differs beyond rounding tolerance. All reference rows included below passed these checks. This verifies our rescoring of the published logs; it does not independently reproduce the reference systems' inference runs.

Two reported quantities are our own calculations rather than values copied from the board:

- **95% confidence intervals:** we calculated these using an item-cluster bootstrap with 2,000 resamples and random seed `20260918`. They can differ from the published intervals and aren't board-checked.
- **Decisions/s:** we computed this as the number of decisions divided by the summed decision time in seconds, pulled directly from the run logs. It isn't a published board metric, nor does it represent wall-clock throughput under concurrent load.

## Timing and interpretation limits

Reference latency comes straight from the published logs, where Jevals measured those requests over the internet at concurrency 4. Local latency comes from our own runs, processing one request at a time. Qwen3 0.6B, MiniCPM5 2B and Qwen3.5 4B ran as MLX 8-bit builds on an Apple Silicon Mac. GLM 4.7 Flash ran as a GGUF Q4_K_M build on a second machine, reached through LM Studio's LM Link, so its latency reflects different hardware and includes the hop between the two machines. We didn't remeasure Jev's latency. The scoring code is shared, but the execution conditions aren't, so these tables aren't a controlled speed comparison. We also didn't measure cost or energy use. Point-estimate rankings on their own don't establish statistically significant differences.

## Findings

This interpretation covers the September 18, 2026 release and the local runs shown below. Jev pairs strong decision quality with low reported latency. It scores higher on the Decision Score than every local model across all three tasks, though it doesn't lead every comparison: Gemini 3.8 Flash actually scores higher on PubMedQA and Banking77. These comparisons rely on Jevals' published answers that we rescored, rather than fresh tests of those services.

Each task asks a system to pick an answer and assign probabilities to the possible options. PubMedQA tests medical yes/no decisions, Banking77 tests banking intent classification, and HelpSteer2 tests ratings of response helpfulness. Jev's published probabilities are labelled native. The six reference LLMs verbalize probabilities in their output. For our local models, we compare that approach with readout, which extracts probabilities from next-token log probabilities. The tables compare complete model-and-method combinations; they don't isolate the effect of model size or architecture.

Accuracy shows how often the selected answer is correct. Decision Score evaluates the probability distribution against a baseline that just uses label frequencies without reading the question. Zero means matching that baseline's loss, and a negative score means doing worse. ECE measures the gap between confidence and observed correctness, where lower values indicate better calibration under that metric. Validity only tells us whether the output met the answer format. A system can produce valid answers, or even higher accuracy, while assigning less useful probabilities.

On PubMedQA, Gemini leads with a Decision Score of 73.0 and 92.5% accuracy, compared with Jev's 69.0 and 91.3%. On Banking77, Gemini again leads at 74.1 and 84.6%, versus Jev's 67.8 and 79.7%. GLM-5.3 sits close to Jev on Banking77, with a score of 66.8. Qwen3.8 Flash, DeepSeek V4.1 Flash, Mistral Medium 3.5, and Mercury 2.5 all show lower score point estimates than Jev on these two tasks. Jev is competitive, but these results don't make it the quality leader on every task. The intervals for Jev and Gemini overlap, so we'd need a paired analysis of their score differences to assess statistical significance. Overlap alone establishes neither a difference nor equivalence.

Jev also doesn't have the lowest calibration error. On Banking77, for example, its ECE is 9.8, compared with 2.8 for DeepSeek and 3.2 for Gemini. DeepSeek still carries a lower Decision Score than Jev. Calibration error gives us one useful view of the probabilities, but it doesn't capture everything the decision metric rewards.

HelpSteer2 is the difficult case for everyone. Jev has the highest score point estimate at 9.2, followed by GLM-5.3 at 7.8, GLM 4.7 Flash with readout at 4.8 and Gemini at 4.6. The 95% intervals of Jev, GLM-5.3 and Gemini include zero. GLM 4.7 Flash with readout is the only system whose interval lies entirely above the label-prior baseline on this task, and only just, at [0.2, 9.1]. Jev's lead in point estimate therefore isn't a clear demonstration that it adds value over that baseline.

Qwen3.5 4B with readout is the strongest local combination by Decision Score on PubMedQA and Banking77. It reaches 45.3 on PubMedQA and 51.9 on Banking77, compared with Jev's 69.0 and 67.8. Its corresponding accuracies are 82.1% and 69.9%, versus Jev's 91.3% and 79.7%. It serves as useful evidence that a small local model can beat the label-prior baseline on these tasks, while still leaving a substantial gap to Jev's published results.

GLM 4.7 Flash is the largest local model, at 30B parameters, but it doesn't beat Qwen3.5 4B on those two tasks. With readout it scores 21.6 on PubMedQA and 43.2 on Banking77, with accuracies of 79.0% and 61.5%, and its PubMedQA probabilities are less well calibrated (ECE 19.1 against Qwen's 4.9). On HelpSteer2 it is the strongest local combination, at 4.8 against Qwen's -11.4. These results cover GLM as configured here: a 4-bit GGUF build, with its default reasoning switched off by the empty think block that opens every answer.

GLM 4.7 Flash's Banking77 readout row isn't a like-for-like comparison with the other local readout rows. GLM reads each two-digit label as a single token, and LM Studio returns only the 10 most likely tokens, so GLM gives a probability to at most 10 of the 77 options: 6.1 on average, against 67.3 for Qwen3.5 4B, which reads the labels one digit at a time. Every other option gets probability 0.

On HelpSteer2, Qwen 4B readout actually has the highest accuracy in the table, at 44.1%, ahead of Jev's 41.3%. Yet its Decision Score is -11.4, compared with Jev's 9.2. GLM 4.7 Flash readout shows the reverse: its accuracy of 38.3% is below the label prior's 41.7%, yet its Decision Score is positive, and its ECE of 9.7 is the lowest on this task. These are the clearest examples of why choosing a system on accuracy alone can be misleading when downstream decisions depend on its probabilities.

The two smallest models are less convincing. Qwen3 0.6B has negative Decision Scores on every task with both methods. MiniCPM5 2B has positive scores on Banking77, but negative point estimates on PubMedQA and HelpSteer2. Those results describe these models and configurations; they don't establish a universal minimum model size for decision tasks.

For Qwen 4B, readout improves Decision Score over verbalized output on every task: 19.0 to 45.3 on PubMedQA, -37.4 to -11.4 on HelpSteer2, and 44.9 to 51.9 on Banking77. The same holds for GLM 4.7 Flash: 6.9 to 21.6, -38.2 to 4.8, and 29.1 to 43.2. Readout also gives valid outputs for every local model on every task. That formatting reliability is useful, but it isn't a guarantee of good probabilities.

MiniCPM5 shows why readout isn't an automatic improvement. On HelpSteer2 it raises accuracy from 31.0% to 41.3%, while lowering Decision Score from -39.3 to -60.7. On Banking77 it also raises accuracy while lowering the score. The model, task, and extraction method need to be evaluated together.

In the published logs, Jev has the lowest median and p95 latency among the reference systems on every task. Its median latency ranges from 438 to 478 ms, whereas Gemini's ranges from 1,636 to 1,887 ms. These logs show an attractive quality-and-latency balance for Jev, even where Gemini has higher quality point estimates. They describe the published measurement conditions, not a service guarantee or a speed measurement we independently repeated.

The local results add another tradeoff. Qwen 4B readout has median latencies of 526 ms on PubMedQA and 370 ms on HelpSteer2, but 4,185 ms on Banking77. On that last task, verbalized output takes 1,924 ms: readout's better score comes with more than twice the median latency. The tables alone don't establish the cause of this slowdown. GLM 4.7 Flash has the lowest median latency in every table, 54 to 303 ms with readout, but it ran on different hardware from the other local models, so its timings aren't comparable with theirs. Local and published timings use different execution conditions, so they can't establish a controlled speed advantage.

Running locally gives control over where inference happens and which model is served, at the cost of managing the hardware and accepting the quality and latency measured for that setup. This benchmark doesn't establish a cost, energy, or privacy-compliance advantage. For these tasks, Jev's published results set a stronger decision-quality reference than our local models; Qwen 4B readout is the strongest local candidate we tested on PubMedQA and Banking77, and GLM 4.7 Flash readout on HelpSteer2. A deployment decision should still check the actual task's probabilities, error consequences, and latency requirements.

## pubmedqa (noul)

| System | Probabilities | Decision Score [95% CI] | Accuracy | ECE | Valid | p50 ms | p95 ms | Decisions/s |
|---|---|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | verbalized | 73.0 [63.2, 81.9] | 92.5% | 2.0 | 100.0% | 1854 | 5538 | 0.41 |
| Jev | native | 69.0 [60.5, 76.7] | 91.3% | 5.0 | 100.0% | 438 | 653 | 2.02 |
| Qwen3.8 Flash | verbalized | 62.4 [51.6, 73.0] | 89.7% | 2.5 | 100.0% | 1088 | 3515 | 0.69 |
| GLM-5.3 | verbalized | 60.6 [50.4, 69.7] | 88.7% | 2.0 | 100.0% | 1757 | 2878 | 0.53 |
| Mistral Medium 3.5 | verbalized | 58.0 [45.9, 69.4] | 88.8% | 5.2 | 100.0% | 533 | 917 | 1.72 |
| Mercury 2.5 | verbalized | 55.7 [45.1, 65.6] | 87.1% | 4.7 | 100.0% | 584 | 1036 | 1.51 |
| DeepSeek V4.1 Flash | verbalized | 47.5 [35.0, 59.1] | 83.7% | 6.2 | 100.0% | 838 | 1123 | 1.16 |
| Qwen3.5 4B (local) | readout | 45.3 [35.9, 54.6] | 82.1% | 4.9 | 100.0% | 526 | 1043 | 1.80 |
| GLM 4.7 Flash (local) | readout | 21.6 [15.1, 27.2] | 79.0% | 19.1 | 100.0% | 58 | 213 | 11.51 |
| Qwen3.5 4B (local) | verbalized | 19.0 [4.3, 33.8] | 68.1% | 17.0 | 100.0% | 1011 | 1755 | 0.92 |
| GLM 4.7 Flash (local) | verbalized | 6.9 [-5.0, 18.5] | 74.3% | 18.9 | 99.9% | 220 | 2069 | 2.27 |
| Label prior | base rates | 0.0 [0.0, 0.0] | 62.0% | — | 100.0% | — | — | — |
| MiniCPM5 2B (local) | readout | -3.0 [-14.0, 9.2] | 66.0% | 22.6 | 100.0% | 480 | 788 | 1.92 |
| Qwen3 0.6B (local) | readout | -28.8 [-40.0, -16.2] | 62.3% | 30.4 | 100.0% | 415 | 585 | 2.35 |
| Qwen3 0.6B (local) | verbalized | -29.9 [-45.4, -15.8] | 56.9% | 28.1 | 100.0% | 833 | 1169 | 1.20 |
| MiniCPM5 2B (local) | verbalized | -66.6 [-93.7, -43.0] | 50.5% | 40.5 | 100.0% | 1048 | 1486 | 0.92 |

## helpsteer2 (score)

| System | Probabilities | Decision Score [95% CI] | Accuracy | ECE | Valid | p50 ms | p95 ms | Decisions/s |
|---|---|---|---|---|---|---|---|---|
| Jev | native | 9.2 [-4.5, 21.4] | 41.3% | 19.7 | 100.0% | 478 | 670 | 1.99 |
| GLM-5.3 | verbalized | 7.8 [-5.4, 19.5] | 43.0% | 12.9 | 100.0% | 2211 | 3631 | 0.43 |
| GLM 4.7 Flash (local) | readout | 4.8 [0.2, 9.1] | 38.3% | 9.7 | 100.0% | 54 | 332 | 10.20 |
| Gemini 3.8 Flash | verbalized | 4.6 [-12.6, 20.2] | 42.4% | 22.6 | 100.0% | 1887 | 3979 | 0.47 |
| Label prior | base rates | 0.0 [0.0, 0.0] | 41.7% | — | 100.0% | — | — | — |
| Qwen3.8 Flash | verbalized | -1.4 [-16.1, 12.1] | 36.1% | 25.8 | 100.0% | 1422 | 2638 | 0.58 |
| Mercury 2.5 | verbalized | -5.5 [-16.2, 4.7] | 41.7% | 31.5 | 100.0% | 618 | 1164 | 1.42 |
| Qwen3.5 4B (local) | readout | -11.4 [-25.1, 1.6] | 44.1% | 29.3 | 100.0% | 370 | 1282 | 2.13 |
| Mistral Medium 3.5 | verbalized | -13.7 [-27.8, -0.2] | 43.9% | 33.6 | 100.0% | 706 | 1178 | 1.31 |
| DeepSeek V4.1 Flash | verbalized | -19.0 [-37.6, -3.0] | 34.7% | 33.5 | 100.0% | 912 | 1203 | 1.07 |
| Qwen3.5 4B (local) | verbalized | -37.4 [-57.7, -19.0] | 33.5% | 30.5 | 99.9% | 1491 | 2852 | 0.62 |
| GLM 4.7 Flash (local) | verbalized | -38.2 [-44.5, -31.9] | 37.7% | 45.6 | 100.0% | 421 | 1612 | 1.80 |
| MiniCPM5 2B (local) | verbalized | -39.3 [-44.6, -33.9] | 31.0% | 53.1 | 98.9% | 932 | 1820 | 0.94 |
| Qwen3 0.6B (local) | readout | -40.5 [-46.1, -35.0] | 30.0% | 66.2 | 100.0% | 404 | 635 | 2.54 |
| MiniCPM5 2B (local) | readout | -60.7 [-71.7, -49.2] | 41.3% | 55.0 | 100.0% | 414 | 887 | 2.11 |
| Qwen3 0.6B (local) | verbalized | -60.8 [-77.0, -47.1] | 16.2% | 28.8 | 99.3% | 1084 | 1673 | 0.92 |

## banking77 (choice)

| System | Probabilities | Decision Score [95% CI] | Accuracy | ECE | Valid | p50 ms | p95 ms | Decisions/s |
|---|---|---|---|---|---|---|---|---|
| Gemini 3.8 Flash | verbalized | 74.1 [68.0, 80.1] | 84.6% | 3.2 | 100.0% | 1636 | 3938 | 0.51 |
| Jev | native | 67.8 [61.0, 74.4] | 79.7% | 9.8 | 100.0% | 467 | 693 | 1.96 |
| GLM-5.3 | verbalized | 66.8 [61.9, 72.0] | 78.9% | 6.2 | 100.0% | 2110 | 3186 | 0.45 |
| DeepSeek V4.1 Flash | verbalized | 63.9 [58.5, 69.4] | 75.9% | 2.8 | 99.9% | 999 | 1309 | 0.97 |
| Qwen3.8 Flash | verbalized | 62.4 [56.1, 69.5] | 76.8% | 9.5 | 99.9% | 1713 | 3105 | 0.51 |
| Mistral Medium 3.5 | verbalized | 59.5 [54.1, 64.8] | 74.5% | 8.1 | 99.5% | 936 | 1444 | 1.00 |
| Mercury 2.5 | verbalized | 54.0 [47.8, 60.7] | 70.3% | 10.4 | 99.7% | 638 | 1478 | 1.26 |
| Qwen3.5 4B (local) | readout | 51.9 [44.9, 59.0] | 69.9% | 16.7 | 100.0% | 4185 | 4605 | 0.24 |
| Qwen3.5 4B (local) | verbalized | 44.9 [40.3, 49.7] | 67.5% | 15.3 | 99.3% | 1924 | 2532 | 0.53 |
| GLM 4.7 Flash (local) | readout | 43.2 [37.0, 49.8] | 61.5% | 12.2 | 100.0% | 303 | 375 | 3.68 |
| GLM 4.7 Flash (local) | verbalized | 29.1 [21.8, 36.1] | 57.2% | 26.2 | 99.2% | 669 | 2563 | 1.14 |
| MiniCPM5 2B (local) | verbalized | 26.9 [21.6, 32.5] | 46.9% | 12.1 | 98.0% | 1536 | 2272 | 0.64 |
| MiniCPM5 2B (local) | readout | 23.5 [16.5, 31.1] | 51.5% | 28.6 | 100.0% | 586 | 814 | 1.80 |
| Label prior | base rates | 0.0 [0.0, 0.0] | 1.3% | — | 100.0% | — | — | — |
| Qwen3 0.6B (local) | verbalized | -2.8 [-5.5, -0.2] | 16.7% | 20.1 | 57.5% | 971 | 2848 | 0.55 |
| Qwen3 0.6B (local) | readout | -33.6 [-36.4, -30.7] | 9.2% | 54.7 | 100.0% | 1418 | 1894 | 0.74 |
