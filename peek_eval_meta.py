# peek_eval_meta.py
# Usage:
#   python peek_eval_meta.py path\to\one.eval
#   python peek_eval_meta.py logs\*.eval
import sys, json, zipfile, os

def peek(path):
    with zipfile.ZipFile(path, "r") as z:
        start = json.loads(z.read("_journal/start.json").decode("utf-8"))
    ev = start.get("eval", {}) or {}
    cat = (ev.get("task_attribs") or {}).get("category")
    task = ev.get("task") or ev.get("task_registry_name")
    created = ev.get("created")
    print(f"{os.path.basename(path)} | task={task} | category={cat} | created={created}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python peek_eval_meta.py <file_or_glob>")
        sys.exit(1)
    for arg in sys.argv[1:]:
        for p in sorted(os.path.abspath(p) for p in __import__("glob").glob(arg)):
            try:
                peek(p)
            except Exception as e:
                print(f"{os.path.basename(p)} | ERROR: {e}")
