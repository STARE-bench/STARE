import argparse
import logging
import os
import json
from collections import defaultdict


def gen_score(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    total_correct = 0
    total_count = 0

    # Only track overall and per‐category accuracy
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for key, entry in data.items():
        total_count += 1
        is_correct = 1 if entry.get("true_false", False) else 0
        total_correct += is_correct

        category = entry.get("category", "Unknown")
        category_stats[category]["total"] += 1
        category_stats[category]["correct"] += is_correct

    average_accuracy = total_correct / total_count if total_count > 0 else 0
    logging.info(f"Average accuracy: {average_accuracy}")

    score = {
        "average": {
            "accuracy": average_accuracy,
            "correct": total_correct,
            "total": total_count
        },
        "category": {
            category: {
                "accuracy": stats["correct"] / stats["total"]
                            if stats["total"] > 0 else 0,
                "correct": stats["correct"],
                "total": stats["total"]
            }
            for category, stats in category_stats.items()
        }
    }

    with open(output_file, "w") as f:
        f.write(json.dumps(score, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='')
    args = parser.parse_args()

    for root, dirs, files in os.walk(args.results_dir):
        for file in files:
            if file.endswith(".json") and not file.endswith("_result.json"):
                input_path = os.path.join(root, file)
                output_path = input_path.replace('.json', '_result.json')
                gen_score(input_path, output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]"
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
