import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def safe_key(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_prompt_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompts(config):
    prompts = {}
    for key, text in config.get("aliases", {}).items():
        prompts[safe_key(key)] = text

    template = config.get("template", "{water_type}, {environment}")
    for water_key, water_text in config["water_types"].items():
        prompts[safe_key(water_key)] = water_text
        for env_key, env_text in config["environments"].items():
            prompts[safe_key(f"{water_key}_{env_key}")] = template.format(
                water_type=water_text,
                environment=env_text,
            )
    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate BERT-base-uncased token features for DirectSAM-RS water prompts."
    )
    parser.add_argument("--config", default="water_prompts.json")
    parser.add_argument("--output", default="water_bert_text_feature.pth")
    parser.add_argument("--manifest", default="water_bert_text_feature_manifest.json")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    prompts = build_prompts(load_prompt_config(args.config))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text_encoder = AutoModel.from_pretrained(args.model).to(args.device)
    text_encoder.eval()

    feature_dict = {}
    manifest = {
        "model": args.model,
        "context_length": args.context_length,
        "prompts": {},
        "note": "Token-level last_hidden_state features from BERT-base-uncased."
    }

    with torch.no_grad():
        for key, prompt in prompts.items():
            encoded = tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=args.context_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(args.device) for k, v in encoded.items()}
            outputs = text_encoder(**encoded)
            feature_dict[f"{key}_feature"] = outputs.last_hidden_state.cpu().float()
            feature_dict[f"{key}_mask"] = encoded["attention_mask"].cpu().long()
            manifest["prompts"][key] = prompt

    torch.save(feature_dict, args.output)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(prompts)} BERT prompt features to {Path(args.output)}")
    print(f"Saved prompt manifest to {Path(args.manifest)}")


if __name__ == "__main__":
    main()
