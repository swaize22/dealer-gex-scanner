name: Weekly Precious Metals COT & Options Chart

on:
  schedule:
    # Runs every Saturday at 15:00 UTC
    - cron: '0 15 * * 6'
  workflow_dispatch: # Allows manual run anytime

permissions:
  contents: write # Allows GitHub to save the new weekly PCR row back into your CSV file

jobs:
  update-metals-chart:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install yfinance numpy pandas requests plotly

      - name: Run Metals & COT Terminal
        run: python metals_macro_terminal.py

      - name: Save Updated ETF Put/Call History CSV to Repo
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add metals_etf_pcr_history.csv
          git diff --staged --quiet || git commit -m "Auto-append weekly GLD/SLV/GDX Put/Call data"
          git push

      - name: Upload Interactive HTML Chart
        uses: actions/upload-artifact@v4
        with:
          name: precious-metals-dashboard
          path: |
            precious_metals_cot_options_dashboard.html
            metals_etf_pcr_history.csv
