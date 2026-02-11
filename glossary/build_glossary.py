name: Build medical glossary (DE) from transcripts

on:
  schedule:
    - cron: "25 3 * * 1"   # every Monday
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout this repo
        uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install deps
        run: |
          pip install -r glossary/requirements.txt

      - name: Build glossary
        env:
          PYTHONUNBUFFERED: "1"
        run: |
          python -u glossary/build_glossary.py \
            --transcripts_dir transcripts \
            --out_csv public/glossary.csv \
            --out_json public/glossary.json \
            --cache_json public/wikidata_cache.json \
            --manual_overrides glossary/manual_overrides.csv \
            --blacklist_path glossary/blacklist.txt \
            --out_todo_csv public/glossary.todo.csv \
            --out_blacklist_suggestions_csv public/blacklist_suggestions.csv \
            --todo_n 20 \
            --blacklist_suggestions_n 50 \
            --max_terms 300 \
            --min_count 2 \
            --max_ngram 4 \
            --sleep_s 0.2

      - name: Commit & push if changed
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add -A \
            public/glossary.csv \
            public/glossary.json \
            public/wikidata_cache.json \
            public/glossary.todo.csv \
            public/blacklist_suggestions.csv
          if git diff --cached --quiet; then
            echo "No changes."
            exit 0
          fi
          git commit -m "Update medical glossary + todo + blacklist suggestions"
          git push
