# run_tests.py  (Step 13 leftover)
# Run:  python scripts/run_tests.py
# Runs pytest and saves the full output to reports/test_results.txt (UTF-8, works on Windows too).
# Why a script? In PowerShell 5 the  >  sign saves files as UTF-16, which many viewers cannot read.

import datetime
import os
import subprocess
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(root, "reports", "test_results.txt")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

result = subprocess.run([sys.executable, "-m", "pytest", "-v", "--tb=short", "-p", "no:cacheprovider"],
                        cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
text = result.stdout + result.stderr

with open(out_path, "w", encoding="utf-8") as f:
    f.write("Test run: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + "\n")
    f.write("Command: python -m pytest -v\n\n")
    f.write(text)

print(text.strip().splitlines()[-1] if text.strip() else "(no output)")
print("Saved:", out_path)
sys.exit(result.returncode)
