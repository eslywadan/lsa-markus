#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert corpus text files to JSON format
Supports command-line arguments or converts all corpus files by default
"""

import json
import re
import sys
from pathlib import Path

def parse_corpus_file(file_path):
    """Parse the corpus file and return structured data"""

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    news_items = []
    current_item = None
    content_lines = []

    for line in lines:
        line = line.strip()

        # Check if this is a date/title line (starts with YYYY/MM/DD)
        date_match = re.match(r'^(\d{4}/\d{2}/\d{2})，<([^>]+)>', line)

        if date_match:
            # Save previous item if exists
            if current_item:
                current_item['news_content'] = '\n'.join(content_lines).strip()
                if current_item['news_content']:  # Only add if has content
                    news_items.append(current_item)

            # Start new item
            date_str = date_match.group(1)
            title = date_match.group(2)

            current_item = {
                'news_date': date_str,
                'news_title': title,
                'news_content': ''
            }
            content_lines = []

        elif line and current_item:
            # This is content for the current news item
            content_lines.append(line)

    # Don't forget the last item
    if current_item:
        current_item['news_content'] = '\n'.join(content_lines).strip()
        if current_item['news_content']:
            news_items.append(current_item)

    return news_items

def convert_file(input_file, output_file):
    """Convert a single corpus file to JSON"""
    print(f"\nReading from {input_file}...")
    news_data = parse_corpus_file(input_file)

    print(f"Found {len(news_data)} news items")

    # Write to JSON file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(news_data, f, ensure_ascii=False, indent=2)

    print(f"Successfully written to {output_file}")

    # Print sample of first item
    if news_data:
        print("\nSample of first news item:")
        print(json.dumps(news_data[0], ensure_ascii=False, indent=2))

    return len(news_data)

def main():
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    corpus_dir = project_root / 'corpus'

    # Check if a specific file was provided as argument
    if len(sys.argv) > 1:
        # Convert specific file
        filename = sys.argv[1]
        if not filename.endswith('.txt'):
            filename += '.txt'

        input_file = corpus_dir / filename
        output_file = corpus_dir / filename.replace('.txt', '.json')

        if not input_file.exists():
            print(f"Error: File {input_file} not found!")
            return

        convert_file(input_file, output_file)
    else:
        # Convert all .txt files in corpus directory
        txt_files = list(corpus_dir.glob('*.txt'))

        if not txt_files:
            print("No .txt files found in corpus directory!")
            return

        print(f"Found {len(txt_files)} corpus file(s) to convert")
        total_items = 0

        for txt_file in sorted(txt_files):
            output_file = txt_file.with_suffix('.json')
            count = convert_file(txt_file, output_file)
            total_items += count

        print(f"\n{'='*60}")
        print(f"Conversion complete! Total news items: {total_items}")
        print(f"{'='*60}")

if __name__ == '__main__':
    main()
