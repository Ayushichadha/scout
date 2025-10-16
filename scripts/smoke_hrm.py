import importlib.util
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
for name in ("hrm", "HRM"):
    if importlib.util.find_spec(name):
        print(f"HRM module found: {name}")
        break
else:
    raise SystemExit("Can't import HRM. Try: pip install -e ./HRM")
print("✅ Import OK")
