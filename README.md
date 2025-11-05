## Finetuning Llasa-1B with GRPO

This repository fine-tunes the Llasa TTS model with GRPO using Hugging Face `transformers`, `trl`, and `datasets`, and evaluates rewards via Whisper ASR and WER.

### Models
- **Llasa**: [HKUSTAudio/Llasa-1B](https://huggingface.co/HKUSTAudio/Llasa-1B)
- **Llasa finetuned with GRPO**: [HKUSTAudio/Llasa-1B](https://huggingface.co/HKUSTAudio/Llasa-1B)
- **Neural codec (decode)**: [HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2)
- **ASR reward model**: `openai/whisper-large-v3`

## Installation

### Step 1: Clone the repository
```bash
git clone https://github.com/your-org/GRPO_Llasa.git
cd GRPO_Llasa
```

### Step 2: Set up environment
Choose your preferred package manager:

<details>
<summary>📦 Using UV (recommended)</summary>

Install `uv` from Astral docs, then:

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install --no-deps xcodec2
```

</details>

<details>
<summary>🐍 Using pip</summary>

```bash
python -m venv .venv --python 3.12 && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps xcodec2
```

</details>

Notes:
- The model was trained on a single A100 80GB GPU.

## Dataset Preparation

GRPO training uses text prompts and computes rewards from generated audio via ASR. The default training script loads:

- `Steveeeeeeen/Elise-xcodec2` (see `create_dataset.py` for how it was built)

Minimum required field per example:

```python
{
  "text": "reference text to be spoken"
}
```

The training script (`train.py`) converts each row into a chat-style prompt:
- User: `Convert the text to speech: <TEXT>`
- Assistant bootstrap: `<|SPEECH_GENERATION_START|>`

Optional fields produced by `create_dataset.py` (not required for GRPO, but useful elsewhere):
- `audio_code_ids` (List[int])
- `audio_code_tokens` (string like "<|s_123|><|s_456|>...")

To build/publish that dataset yourself, see `create_dataset.py` (encodes audio with XCodec2 and pushes to the Hub).

You can use it like this: 

```bash
python create_dataset.py \
  --dataset-id MrDragonFox/Elise \
  --split train \
  --push-id Steveeeeeeen/Elise-xcodec2 \
  --codec-id HKUSTAudio/xcodec2 \
  --sampling-rate 16000
```

## Training

Run the GRPO trainer:
```bash
python train.py \
  --model-id HKUSTAudio/Llasa-1B \
  --dataset-id Steveeeeeeen/Elise-xcodec2 \
  --dataset-split train \
  --output-dir Llasa-1B-GRPO \
  --save-steps 500 \
  --save-total-limit 3 \
  --max-steps 2000
```

What it does (see `train.py`):
- Loads dataset and builds a `prompt` column for GRPO.
- Uses `HKUSTAudio/Llasa-1B` as the policy model.
- Computes reward with `reward_whisper.py` using:
  - Whisper ASR (`openai/whisper-large-v3`) for WER and NLL
  - XCodec2 to decode generated code tokens into waveform
- Saves checkpoints under `Llasa-1B-GRPO/` every 500 steps (keeps last 3).

Customizing:
- Change dataset/model IDs inside `train.py`.
- Adjust save frequency/limits in `GRPOConfig`.
- Tune reward mixing in `reward_whisper.py` (`lambda_*`, `alpha_*`).
- Enable Weights & Biases by setting `WANDB_PROJECT`/`WANDB_API_KEY` (already in `requirements.txt`).

## Inference

Generate a waveform with the base or fine-tuned checkpoint using `inference.py`:

```bash
python inference.py \
  --llasa-id Llasa-1B-GRPO/checkpoint-2000 \
  --codec-id HKUSTAudio/xcodec2 \
  --text "Hello world from Llasa with GRPO." \
  --output gen_grpo.wav \
  --max-length 2048 \
  --temperature 0.8 \
  --top-p 1.0 \
  --device auto
```

## License and usage
Please review the upstream licenses and usage terms:
- [HKUSTAudio/Llasa-1B](https://huggingface.co/HKUSTAudio/Llasa-1B)
- [HKUSTAudio/xcodec2](https://huggingface.co/HKUSTAudio/xcodec2)
- `openai/whisper-large-v3`
