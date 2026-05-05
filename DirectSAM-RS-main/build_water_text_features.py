import argparse
import json
import re
from pathlib import Path

import torch
import open_clip


def safe_key(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_prompt_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def build_prompts(config):
    prompts = {}

    for key, text in config.get("aliases", {}).items():
        prompts[safe_key(key)] = text

    water_types = config["water_types"]
    environments = config["environments"]
    template = config.get("template", "{water_type}, {environment}")

    for water_key, water_text in water_types.items():
        prompts[safe_key(water_key)] = water_text
        for env_key, env_text in environments.items():
            key = safe_key(f"{water_key}_{env_key}")
            prompts[key] = template.format(water_type=water_text, environment=env_text)

    return prompts


def encode_token_features(model, token_ids):
    context_length = token_ids.shape[1]
    cast_dtype = model.transformer.get_cast_dtype()

    x = model.token_embedding(token_ids).to(cast_dtype)
    x = x + model.positional_embedding[:context_length].to(cast_dtype)

    attn_mask = getattr(model, "attn_mask", None)
    if attn_mask is not None:
        attn_mask = attn_mask[:context_length, :context_length]

    x = model.transformer(x, attn_mask=attn_mask)
    x = model.ln_final(x)
    return x.float()


def main():
    parser = argparse.ArgumentParser(
        description="Generate DirectSAM-RS token-level text features for water type and environment prompts."
    )
    parser.add_argument("--config", default="water_prompts.json", help="Prompt config JSON.")
    parser.add_argument("--output", default="water_text_feature.pth", help="Output .pth path.")
    parser.add_argument("--manifest", default="water_text_feature_manifest.json", help="Prompt manifest JSON.")
    parser.add_argument("--model", default="ViT-L-14", help="OpenCLIP text model name.")
    parser.add_argument("--pretrained", default="openai", help="OpenCLIP pretrained tag.")
    parser.add_argument("--context-length", type=int, default=20, help="Token length expected by DirectSAM-RS.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda.")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_path = Path(args.output)
    manifest_path = Path(args.manifest)

    config = load_prompt_config(config_path)
    prompts = build_prompts(config)

    model = open_clip.create_model(args.model, pretrained=args.pretrained, device=args.device)
    model.eval()

    feature_dict = {}
    manifest = {
        "model": args.model,
        "pretrained": args.pretrained,
        "context_length": args.context_length,
        "prompts": {}
    }

    with torch.no_grad():
        for key, prompt in prompts.items():
            token_ids = open_clip.tokenize([prompt], context_length=args.context_length).to(args.device)
            features = encode_token_features(model, token_ids).cpu()
            mask = token_ids.ne(0).long().cpu()

            feature_dict[f"{key}_feature"] = features
            feature_dict[f"{key}_mask"] = mask
            manifest["prompts"][key] = prompt

    torch.save(feature_dict, output_path)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(prompts)} prompt features to {output_path}")
    print(f"Saved prompt manifest to {manifest_path}")


if __name__ == "__main__":
    main()
