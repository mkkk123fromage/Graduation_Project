import argparse
import csv
import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from model import SegformerWithTextFusion


IMAGE_COLUMNS = ("image", "image_path", "img", "img_path")
LABEL_COLUMNS = ("contour", "contour_path", "label", "label_path", "mask", "mask_path")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
LABEL_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_key(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def resolve_path(root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(root) / path


def find_column(row, candidates):
    for name in candidates:
        if name in row and row[name]:
            return row[name]
    return None


def load_prompt_config(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_prompt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Empty word file: {path}")
    return lines[0]


def build_prompt(row, prompt_config):
    if "prompt" in row and row["prompt"]:
        return row["prompt"]

    water_type = row.get("water_type", "").strip()
    environment = row.get("environment", "").strip()
    if not water_type:
        raise ValueError("CSV row needs either prompt or water_type.")

    if prompt_config is not None:
        water_type = prompt_config.get("water_types", {}).get(water_type, water_type)
        environment = prompt_config.get("environments", {}).get(environment, environment)
        template = prompt_config.get("template", "{water_type}, {environment}")
    else:
        template = "{water_type}, {environment}"

    if environment:
        return template.format(water_type=water_type, environment=environment)
    return water_type


def build_prompt_key(row):
    if "prompt_key" in row and row["prompt_key"]:
        return safe_key(row["prompt_key"])
    water_type = row.get("water_type", "").strip()
    environment = row.get("environment", "").strip()
    if environment:
        return safe_key(f"{water_type}_{environment}")
    return safe_key(water_type)


class WaterContourDataset(Dataset):
    def __init__(
        self,
        csv_path,
        data_root,
        image_processor,
        resolution,
        text_mode,
        text_features=None,
        tokenizer=None,
        prompt_config=None,
        label_threshold=127,
    ):
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.image_processor = image_processor
        self.resolution = resolution
        self.text_mode = text_mode
        self.text_features = text_features
        self.tokenizer = tokenizer
        self.prompt_config = prompt_config
        self.label_threshold = label_threshold

        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"No rows found in {self.csv_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        image_value = find_column(row, IMAGE_COLUMNS)
        label_value = find_column(row, LABEL_COLUMNS)
        if image_value is None or label_value is None:
            raise ValueError(f"CSV must contain image and contour/label columns. Bad row: {row}")

        image_path = resolve_path(self.data_root, image_value)
        label_path = resolve_path(self.data_root, label_value)

        image = Image.open(image_path).convert("RGB")
        encoding = self.image_processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.squeeze(0)

        label = Image.open(label_path).convert("L")
        label = label.resize((self.resolution, self.resolution), Image.Resampling.NEAREST)
        label = (np.array(label) > self.label_threshold).astype(np.float32)
        labels = torch.from_numpy(label)

        sample = {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_path": str(image_path),
        }

        if self.text_mode == "precomputed":
            prompt_key = build_prompt_key(row)
            feature_key = f"{prompt_key}_feature"
            mask_key = f"{prompt_key}_mask"
            if feature_key not in self.text_features or mask_key not in self.text_features:
                available = sorted(k[:-8] for k in self.text_features if k.endswith("_feature"))
                raise KeyError(f"Prompt key '{prompt_key}' not found. Available examples: {available[:20]}")
            sample["local_text_feature"] = self.text_features[feature_key].squeeze(0).float()
            sample["word_mask"] = self.text_features[mask_key].squeeze(0).long()
        elif self.text_mode == "bert":
            prompt = build_prompt(row, self.prompt_config)
            encoded = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=20,
                return_tensors="pt",
            )
            sample["input_ids"] = encoded["input_ids"].squeeze(0).long()
            sample["token_type_ids"] = encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).squeeze(0).long()
            sample["word_mask"] = encoded["attention_mask"].squeeze(0).long()
        else:
            raise ValueError(f"Unsupported text mode: {self.text_mode}")

        return sample


class TripletDirWaterDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        image_processor,
        resolution,
        text_mode,
        text_features=None,
        tokenizer=None,
        split_file=None,
        images_dir="images",
        words_dir="words",
        labels_dir=None,
        label_threshold=127,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.image_processor = image_processor
        self.resolution = resolution
        self.text_mode = text_mode
        self.text_features = text_features
        self.tokenizer = tokenizer
        self.label_threshold = label_threshold

        self.images_dir = self.dataset_dir / images_dir
        self.words_dir = self.dataset_dir / words_dir
        if labels_dir is None:
            if (self.dataset_dir / "labels").is_dir():
                labels_dir = "labels"
            elif (self.dataset_dir / "lables").is_dir():
                labels_dir = "lables"
            else:
                labels_dir = "labels"
        self.labels_dir = self.dataset_dir / labels_dir

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Missing images directory: {self.images_dir}")
        if not self.words_dir.is_dir():
            raise FileNotFoundError(f"Missing words directory: {self.words_dir}")
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(f"Missing labels directory: {self.labels_dir}")

        split_stems = None
        if split_file:
            split_path = Path(split_file)
            if not split_path.is_absolute():
                split_path = self.dataset_dir / split_path
            with open(split_path, "r", encoding="utf-8") as f:
                split_stems = {Path(line.strip()).stem for line in f if line.strip()}

        self.samples = []
        for image_path in sorted(self.images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if split_stems is not None and image_path.stem not in split_stems:
                continue
            word_path = self.words_dir / f"{image_path.stem}.txt"
            label_path = self._find_label(image_path.stem)
            if not word_path.exists():
                raise FileNotFoundError(f"Missing word file for {image_path.name}: {word_path}")
            if label_path is None:
                raise FileNotFoundError(f"Missing label file for {image_path.name} in {self.labels_dir}")
            self.samples.append((image_path, word_path, label_path))

        if not self.samples:
            raise ValueError(f"No samples found in {self.dataset_dir}")

    def _find_label(self, stem):
        for ext in LABEL_EXTENSIONS:
            path = self.labels_dir / f"{stem}{ext}"
            if path.exists():
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, word_path, label_path = self.samples[idx]
        prompt = read_prompt_file(word_path)
        prompt_key = safe_key(prompt)

        image = Image.open(image_path).convert("RGB")
        encoding = self.image_processor(image, return_tensors="pt")
        pixel_values = encoding.pixel_values.squeeze(0)

        label = Image.open(label_path).convert("L")
        label = label.resize((self.resolution, self.resolution), Image.Resampling.NEAREST)
        label = (np.array(label) > self.label_threshold).astype(np.float32)
        labels = torch.from_numpy(label)

        sample = {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_path": str(image_path),
        }

        if self.text_mode == "precomputed":
            feature_key = f"{prompt_key}_feature"
            mask_key = f"{prompt_key}_mask"
            if feature_key not in self.text_features or mask_key not in self.text_features:
                available = sorted(k[:-8] for k in self.text_features if k.endswith("_feature"))
                raise KeyError(
                    f"Prompt '{prompt}' maps to key '{prompt_key}', but it is not in text features. "
                    f"Available examples: {available[:20]}"
                )
            sample["local_text_feature"] = self.text_features[feature_key].squeeze(0).float()
            sample["word_mask"] = self.text_features[mask_key].squeeze(0).long()
        elif self.text_mode == "bert":
            encoded = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=20,
                return_tensors="pt",
            )
            sample["input_ids"] = encoded["input_ids"].squeeze(0).long()
            sample["token_type_ids"] = encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])).squeeze(0).long()
            sample["word_mask"] = encoded["attention_mask"].squeeze(0).long()
        else:
            raise ValueError(f"Unsupported text mode: {self.text_mode}")

        return sample


def collate_batch(features):
    batch = {
        "pixel_values": torch.stack([item["pixel_values"] for item in features]),
        "labels": torch.stack([item["labels"] for item in features]),
        "word_mask": torch.stack([item["word_mask"] for item in features]),
        "image_path": [item["image_path"] for item in features],
    }
    if "local_text_feature" in features[0]:
        batch["local_text_feature"] = torch.stack([item["local_text_feature"] for item in features])
    if "input_ids" in features[0]:
        batch["input_ids"] = torch.stack([item["input_ids"] for item in features])
        batch["token_type_ids"] = torch.stack([item["token_type_ids"] for item in features])
    return batch


def freeze_visual_encoder(model):
    encoder = getattr(model.segformer, "encoder", None)
    if encoder is None:
        raise AttributeError("Could not find model.segformer.encoder to freeze.")
    for param in encoder.patch_embeddings.parameters():
        param.requires_grad = False
    for param in encoder.block.parameters():
        param.requires_grad = False
    for param in encoder.layer_norm.parameters():
        param.requires_grad = False


def make_optimizer(model, text_encoder, args):
    params = [p for p in model.parameters() if p.requires_grad]
    if text_encoder is not None and not args.freeze_text_encoder:
        params.extend([p for p in text_encoder.parameters() if p.requires_grad])
    return AdamW(params, lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)


def make_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prepare_text_features(batch, text_encoder, text_mode, device, freeze_text_encoder):
    if text_mode == "precomputed":
        return batch["local_text_feature"].to(device), batch["word_mask"].to(device)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["word_mask"].to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    if freeze_text_encoder:
        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
    else:
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    return outputs.last_hidden_state.float(), attention_mask


def compute_binary_metrics(logits, labels, threshold):
    probs = torch.sigmoid(
        torch.nn.functional.interpolate(
            logits.float(),
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    )
    pred = probs[:, 0] > threshold
    target = labels > 0.5

    tp = torch.logical_and(pred, target).sum().item()
    fp = torch.logical_and(pred, torch.logical_not(target)).sum().item()
    fn = torch.logical_and(torch.logical_not(pred), target).sum().item()
    tn = torch.logical_and(torch.logical_not(pred), torch.logical_not(target)).sum().item()
    return tp, fp, fn, tn


def run_epoch(model, text_encoder, loader, optimizer, scheduler, scaler, args, device, train):
    model.train(train)
    if text_encoder is not None:
        text_encoder.train(train and not args.freeze_text_encoder)

    total_loss = 0.0
    total_samples = 0
    total_tp = total_fp = total_fn = total_tn = 0
    optimizer.zero_grad(set_to_none=True)

    desc = "Train" if train else "Val"
    iterator = tqdm(loader, desc=desc)
    for step, batch in enumerate(iterator):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        local_text_feature, word_mask = prepare_text_features(
            batch,
            text_encoder,
            args.text_mode,
            device,
            args.freeze_text_encoder,
        )

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                outputs = model(
                    pixel_values=pixel_values,
                    word_mask=word_mask.unsqueeze(1),
                    local_text_feature=local_text_feature.unsqueeze(1),
                    labels=labels,
                )
                loss = outputs.loss
                loss_for_backward = loss / args.grad_accum_steps

            if train:
                if scaler is not None:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

                if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(loader):
                    if args.max_grad_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for group in optimizer.param_groups for p in group["params"]],
                            args.max_grad_norm,
                        )
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = pixel_values.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        tp, fp, fn, tn = compute_binary_metrics(outputs.logits.detach(), labels.detach(), args.metric_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        iou = total_tp / max(1, total_tp + total_fp + total_fn)
        precision = total_tp / max(1, total_tp + total_fp)
        recall = total_tp / max(1, total_tp + total_fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)
        iterator.set_postfix(loss=total_loss / max(1, total_samples), iou=iou, f1=f1)

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)
    iou = total_tp / max(1, total_tp + total_fp + total_fn)
    pa = (total_tp + total_tn) / max(1, total_tp + total_tn + total_fp + total_fn)

    return {
        "loss": total_loss / max(1, total_samples),
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pa": pa,
    }


def save_checkpoint(save_dir, model, image_processor, text_encoder, tokenizer, metrics, name):
    path = Path(save_dir) / name
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    image_processor.save_pretrained(path)
    if text_encoder is not None:
        text_path = path / "text_encoder"
        text_encoder.save_pretrained(text_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(text_path)
    with open(path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train DirectSAM-RS style water mask extraction.")
    parser.add_argument("--dataset-format", choices=["csv", "triplet_dirs"], default="triplet_dirs")
    parser.add_argument("--dataset-dir", default=None, help="Directory containing images, words, labels/lables.")
    parser.add_argument("--train-split", default=None, help="Optional txt file with train sample names for triplet_dirs.")
    parser.add_argument("--val-split", default=None, help="Optional txt file with val sample names for triplet_dirs.")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--words-dir", default="words")
    parser.add_argument("--labels-dir", default=None, help="Defaults to labels, or lables if that directory exists.")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--val-csv", default=None)
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--init-checkpoint", default="weight_directsam_base")
    parser.add_argument("--processor-checkpoint", default=None)
    parser.add_argument("--save-dir", default="outputs/water_directsam_rs")
    parser.add_argument("--prompt-config", default="water_prompts.json")
    parser.add_argument("--text-mode", choices=["precomputed", "bert"], default="precomputed")
    parser.add_argument(
        "--text-features",
        default="water_text_feature.pth",
        help="Offline token features. Default follows the thesis draft: OpenCLIP text features.",
    )
    parser.add_argument("--bert-model", default="bert-base-uncased")
    parser.add_argument("--freeze-text-encoder", action="store_true")
    parser.add_argument("--freeze-visual-encoder", action="store_true")
    parser.add_argument("--resolution", type=int, default=512, help="Default follows the thesis draft: 512 x 512.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--metric-threshold", type=float, default=0.5)
    parser.add_argument("--label-threshold", type=int, default=127)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    processor_checkpoint = args.processor_checkpoint or args.init_checkpoint
    image_processor = AutoImageProcessor.from_pretrained(processor_checkpoint, reduce_labels=True)
    image_processor.size["height"] = args.resolution
    image_processor.size["width"] = args.resolution

    model, loading_info = SegformerWithTextFusion.from_pretrained(
        args.init_checkpoint,
        ignore_mismatched_sizes=True,
        output_loading_info=True,
    )
    print("Missing keys:", len(loading_info.get("missing_keys", [])))
    print("Unexpected keys:", len(loading_info.get("unexpected_keys", [])))

    if args.freeze_visual_encoder:
        freeze_visual_encoder(model)

    model.to(device)

    text_features = None
    text_encoder = None
    tokenizer = None
    prompt_config = load_prompt_config(args.prompt_config) if args.prompt_config else None

    if args.text_mode == "precomputed":
        text_features = torch.load(args.text_features, map_location="cpu")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
        text_encoder = AutoModel.from_pretrained(args.bert_model).to(device)
        if args.freeze_text_encoder:
            text_encoder.eval()
            for param in text_encoder.parameters():
                param.requires_grad = False

    if args.dataset_format == "csv":
        if not args.train_csv:
            raise ValueError("--train-csv is required when --dataset-format csv")
        train_dataset = WaterContourDataset(
            args.train_csv,
            args.data_root,
            image_processor,
            args.resolution,
            args.text_mode,
            text_features=text_features,
            tokenizer=tokenizer,
            prompt_config=prompt_config,
            label_threshold=args.label_threshold,
        )
        val_dataset = None
        if args.val_csv:
            val_dataset = WaterContourDataset(
                args.val_csv,
                args.data_root,
                image_processor,
                args.resolution,
                args.text_mode,
                text_features=text_features,
                tokenizer=tokenizer,
                prompt_config=prompt_config,
                label_threshold=args.label_threshold,
            )
    else:
        dataset_dir = args.dataset_dir or args.data_root
        train_dataset = TripletDirWaterDataset(
            dataset_dir,
            image_processor,
            args.resolution,
            args.text_mode,
            text_features=text_features,
            tokenizer=tokenizer,
            split_file=args.train_split,
            images_dir=args.images_dir,
            words_dir=args.words_dir,
            labels_dir=args.labels_dir,
            label_threshold=args.label_threshold,
        )
        val_dataset = None
        if args.val_split:
            val_dataset = TripletDirWaterDataset(
                dataset_dir,
                image_processor,
                args.resolution,
                args.text_mode,
                text_features=text_features,
                tokenizer=tokenizer,
                split_file=args.val_split,
                images_dir=args.images_dir,
                words_dir=args.words_dir,
                labels_dir=args.labels_dir,
                label_threshold=args.label_threshold,
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_batch,
        )

    optimizer = make_optimizer(model, text_encoder, args)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    scheduler = make_scheduler(optimizer, args.epochs * updates_per_epoch, args.warmup_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    best_score = -1.0
    patience_counter = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_metrics = run_epoch(model, text_encoder, train_loader, optimizer, scheduler, scaler, args, device, True)
        print("train:", train_metrics)

        metrics = {"epoch": epoch, "train": train_metrics}
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(model, text_encoder, val_loader, optimizer, scheduler, scaler, args, device, False)
            print("val:", val_metrics)
            metrics["val"] = val_metrics
            score = val_metrics["iou"]
        else:
            score = train_metrics["iou"]

        history.append(metrics)
        with open(Path(args.save_dir) / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        save_checkpoint(args.save_dir, model, image_processor, text_encoder, tokenizer, metrics, "last")
        if score > best_score + args.early_stopping_min_delta:
            best_score = score
            patience_counter = 0
            save_checkpoint(args.save_dir, model, image_processor, text_encoder, tokenizer, metrics, "best")
        else:
            patience_counter += 1

        if val_loader is not None and args.early_stopping_patience > 0:
            print(f"early stopping patience: {patience_counter}/{args.early_stopping_patience}")
            if patience_counter >= args.early_stopping_patience:
                print("Early stopping triggered.")
                break

    print(f"Best IoU: {best_score:.6f}")


if __name__ == "__main__":
    main()
