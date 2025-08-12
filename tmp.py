#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Generate date_prompt{xx}.json and date_query{xx}.sql (xx = two digits, 00–99)."
    )
    parser.add_argument("start", type=int, help="Start of range (inclusive, 0–99)")
    parser.add_argument("end", type=int, help="End of range (inclusive, 0–99)")
    args = parser.parse_args()

    start, end = args.start, args.end
    if start > end:
        parser.error(f"start ({start}) must be <= end ({end}).")
    if not (0 <= start <= 99 and 0 <= end <= 99):
        parser.error("Both start and end must be between 0 and 99 to keep exactly two digits.")

    now_str = datetime.now().isoformat(timespec="seconds")

    for i in range(start, end + 1):
        xx = f"{i:02d}"  # two digits
        prompt_path = Path(f"date_prompt{xx}.json")
        query_path  = Path(f"date_query{xx}.sql")

        # JSON
        prompt_content = {
            "name": f"date_prompt{xx}",
            "index": i,
            "index_padded": xx,
            "generated_at": now_str
        }
        with prompt_path.open("w", encoding="utf-8") as f:
            json.dump(prompt_content, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # SQL
        query_content = (
            f"-- {query_path.name}\n"
            f"-- Auto-generated at {now_str}\n"
            f"-- Template query for index {i} (xx={xx})\n"
            "SELECT *\n"
            "FROM your_table\n"
            f"WHERE your_column = {i};\n"
        )
        with query_path.open("w", encoding="utf-8") as f:
            f.write(query_content)

        print(f"Generated: {prompt_path} and {query_path}")

if __name__ == "__main__":
    main()