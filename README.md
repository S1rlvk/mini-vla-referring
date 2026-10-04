# mini-vla-referring

Does adding a VLM help a robot policy? A small test in a 2D sim.

A top-down scene has 2 to 4 shapes. A sentence picks out one. A gripper must reach that shape. Touching a decoy or timing out is a failure. There are 4 levels of instruction complexity and 64 scenes per level.

- **Qwen system:** a frozen Qwen3.5-9B (4-bit MLX) reads the image and instruction and outputs target coordinates. A tiny trained MLP head drives the gripper there.
- **Engineered baseline:** a hand-written parser plus a blob tracker, feeding the same head.

L4 ("between", "not red") is held out from head training, not from Qwen's pretraining.

## Results

Successes out of 64.

| Method | Wording | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| Blind-parser tracker | template | 63 | 63 | 61 | 61 |
| Blind-parser tracker | paraphrased | 59 | 48 | 57 | 52 |
| Blind-parser tracker, "wedge" added | paraphrased | 63 | 53 | 57 | 61 |
| Qwen points + same head | template | 41 | 41 | 34 | 45 |
| Qwen points + same head | paraphrased | 43 | 42 | 32 | 40 |

- The tracker wins on accuracy at every level.
- Qwen picks the right object: its first move aims at the target in 252 of 256 episodes and it touches a decoy 1 to 5 times per 64.
- Its limit is precision: median point error 19 px against an 8 to 14 px object radius, mostly noise rather than bias (slope about 1.05).
- The baseline parser was written by a separate agent from the task description, without the test set. One unseen synonym ("wedge" for triangle) cost 18 episodes (216 to 234 of 256 paraphrased). Adding it was a one-line change.
- An earlier variant that fed Qwen's hidden state to the head showed 0.97 plan accuracy at L1, but in counterfactual pairs (same scene, retargeted instruction) it switched targets in only 5 of 32 pairs.

## Caveats

- One run per condition, no repeats. Small differences between levels are noise.
- The scene generator and the paraphrases were written by the same author as the parser's evaluation harness (an AI assistant). The blind parser's brief named the relations, so it was not blind to the grammar.
- The "wedge" fix was chosen after seeing it fail, so 18 episodes is an upper bound on what one word is worth.
- A 12-episode pointer pilot reused test scenes, and the pointer prompt was tuned for output format after viewing it.

## Layout

- `src/cloth_occlusion/`: scenes and reaching task, experiment runner, Qwen pointer (`referring_point.py`), paraphrases, blind-parser runner. The package keeps its name from an earlier benchmark.
- `configs/referring_128.toml`: config for the reported runs. Edit `model_path` for your machine.
- `baselines/blind_parser/` and `baselines/blind_parser_wedge/`: the blind parser and the copy with "wedge" added.
- `results/`: comparison JSONs and per-episode rollouts. Trained heads and feature caches are not included.

## Run

Needs the Qwen weights and `mlx-vlm`.

```bash
uv run --with mlx-vlm python -m cloth_occlusion.referring_experiment --config configs/referring_128.toml --output runs/referring_128
uv run --with mlx-vlm python -m cloth_occlusion.referring_point --config configs/referring_128.toml --heads runs/referring_128 --output runs/referring_point
uv run --with mlx-vlm python -m cloth_occlusion.referring_paraphrase --config configs/referring_128.toml --heads runs/referring_128 --output runs/referring_paraphrase
uv run python -m cloth_occlusion.referring_blind --config configs/referring_128.toml --heads runs/referring_128 --parser_dir baselines/blind_parser --output runs/referring_blind
```
