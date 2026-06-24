#!/usr/bin/env python3
"""sparkinfer PR auto-evaluator (bot).

Polls open PRs; for any PR whose head commit hasn't been evaluated yet, runs the vast.ai
evaluation (build → correctness → speed → label), applies an `eval:<LABEL>` label, and posts the
result as a PR comment. **Never merges** — merging is manual after review.

Designed to run on a 30-min schedule (system cron or a Claude agent). Idempotent: a commit is
evaluated once (tracked by a hidden marker in the bot's comment), so re-runs only pick up new
commits and only spin the GPU when there's new work.

  python eval/pr_eval_bot.py --instance 42134865 --frontier 164 --ceiling 366

Needs: `gh` authenticated, VAST_API_KEY saved (vastai), and the eval:* labels (eval/setup_labels.sh).
"""
import argparse, datetime, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# vast_eval.py self-heals dead boxes (recreates them) and writes the working instance id here;
# prefer it over --instance so we reuse the recreated box instead of retrying the dead one.
INSTANCE_FILE = os.path.expanduser(os.environ.get("VAST_INSTANCE_FILE", "~/.sparkinfer_vast_instance"))
def current_instance(default):
    try: return int(open(INSTANCE_FILE).read().strip())
    except Exception: return default

# Subsystem buckets for the deterministic area:<name> label (from a PR's top-level changed
# dirs — no AI). Categorization/display only: SN74 scoring is speedup-only (the eval:* tier),
# NOT a per-subsystem budget.
AREAS = {"kernels", "runtime", "moe", "bench"}

def gh(args):
    return subprocess.run(["gh"] + args, capture_output=True, text=True)

def evaluated_commits(repo, num):
    r = gh(["pr", "view", str(num), "-R", repo, "--json", "comments"])
    done = set()
    for c in json.loads(r.stdout or "{}").get("comments", []):
        for m in re.finditer(r"<!-- sparkinfer-eval:([0-9a-f]+) -->", c.get("body", "")):
            done.add(m.group(1))
    return done

def areas_for_pr(repo, num):
    """Subsystems a PR touches, from its changed file paths (deterministic, no AI)."""
    files = json.loads(gh(["pr", "view", str(num), "-R", repo, "--json", "files"]).stdout or "{}").get("files", [])
    return {f["path"].split("/", 1)[0] for f in files} & set(AREAS)

def _owner_repo(repo):
    parts = repo.split("/"); return parts[0], parts[1]

def labels_on(repo, num):
    owner, r = _owner_repo(repo)
    out = subprocess.run(["gh", "api", f"repos/{owner}/{r}/issues/{num}/labels",
                          "--jq", "[.[].name]"], capture_output=True, text=True)
    try: return set(json.loads(out.stdout))
    except Exception: return set()

def add_label(repo, num, label):
    owner, r = _owner_repo(repo)
    subprocess.run(["gh", "api", f"repos/{owner}/{r}/issues/{num}/labels",
                    "--method", "POST", "-f", f"labels[]={label}"],
                   capture_output=True, text=True)

def remove_label(repo, num, label):
    owner, r = _owner_repo(repo)
    subprocess.run(["gh", "api", f"repos/{owner}/{r}/issues/{num}/labels/{label}",
                    "--method", "DELETE"], capture_output=True, text=True)

def apply_area_labels(repo, num, areas):
    want = {f"area:{a}" for a in areas}
    have = {l for l in labels_on(repo, num) if l.startswith("area:")}
    for lab in want - have: add_label(repo, num, lab)
    for lab in have - want: remove_label(repo, num, lab)

def render(res, oid):
    label = res.get("label", "?")
    icon = {"REJECT": "❌", "none": "⚪", "BASELINE": "📊"}.get(label, "✅")
    rows = [f"| **label** | `eval:{label}` |",
            f"| decode | {res.get('tps','?')} tok/s |",
            f"| correctness | top-1 {res.get('top1',0)*100:.1f}% · KL {res.get('kl','?')} |"]
    if "frontier_tps" in res and res["frontier_tps"]:
        rows.insert(2, f"| vs frontier | {res['frontier_tps']} tok/s → "
                       f"{res.get('pct_over_frontier', 0):+.1f}% ({res.get('delta_tps',0):+.1f}) |")
    note = {"REJECT": f"Failed the correctness gate: {res.get('reason','')}. Not a valid submission.",
            "none": "Within the significance gate — no *verified* speedup over the current frontier.",
            "BASELINE": "No frontier was set; this run establishes it."
            }.get(label, "Verified speedup over the live frontier.")
    return (f"<!-- sparkinfer-eval:{oid} -->\n"
            f"## {icon} sparkinfer auto-eval — `{oid}`\n\n"
            f"| metric | value |\n|---|---|\n" + "\n".join(rows) + "\n\n"
            f"{note}\n\n"
            f"_RTX 5090 (sm_120) · built from source · correctness vs llama.cpp. "
            f"Automated — **not merged**; merge manually after review._")

# ---- live dashboard: data.json is canonical; data.js is generated for the page ----
DASH = os.path.join(ROOT, "dashboard")
DATA_JSON = os.path.join(DASH, "data.json")
FRONTIER_LABELS = {"XL", "L", "M", "S", "XS", "BASELINE"}

def load_dash():
    try:
        with open(DATA_JSON) as f: return json.load(f)
    except Exception:
        return None

def write_dash(data):
    with open(DATA_JSON, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    with open(os.path.join(DASH, "data.js"), "w") as f:
        f.write("// Generated by eval/pr_eval_bot.py from data.json — do not edit by hand.\n")
        f.write("window.SPARKINFER = " + json.dumps(data, indent=2, ensure_ascii=False) + ";\n")

def push_dash(msg):
    subprocess.run(["git", "-C", ROOT, "add", "dashboard/data.json", "dashboard/data.js"], capture_output=True)
    if subprocess.run(["git", "-C", ROOT, "diff", "--cached", "--quiet"]).returncode == 0:
        return  # nothing changed
    subprocess.run(["git", "-C", ROOT, "commit", "-q", "-m", msg], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "pull", "-q", "--rebase", "origin", "main"], capture_output=True)
    subprocess.run(["git", "-C", ROOT, "push", "-q", "origin", "main"], capture_output=True)

def update_dashboard(repo, pr, areas, res):
    """Upsert the PR's verdict into the dashboard, ratchet the frontier, regenerate + push."""
    data = load_dash()
    if data is None: return
    num = pr["number"]
    entry = {"num": num, "title": pr.get("title", ""), "areas": sorted(areas),
             "label": res.get("label"), "tps": res.get("tps"),
             "delta_pct": res.get("pct_over_frontier"),
             "url": f"https://github.com/{repo}/pull/{num}"}
    data["prs"] = [p for p in data.get("prs", []) if p.get("num") != num]
    data["prs"].insert(0, entry)
    data["prs"] = data["prs"][:50]
    if (res.get("pass") and res.get("label") in FRONTIER_LABELS
            and (res.get("tps") or 0) > data["status"].get("frontier_tps", 0)):
        data["status"]["frontier_tps"] = res["tps"]            # ratchet the live frontier
    data["updated"] = datetime.date.today().isoformat()
    write_dash(data)
    push_dash(f"dashboard: PR #{num} -> eval:{res.get('label')} ({res.get('tps')} tok/s)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", type=int, required=True, help="vast.ai instance id to reuse")
    ap.add_argument("--frontier", type=float, default=0)
    ap.add_argument("--ceiling", type=float, default=0)
    ap.add_argument("--repo", default="gittensor-ai-lab/sparkinfer")
    ap.add_argument("--dry-run", action="store_true", help="evaluate + print, but don't label/comment")
    args = ap.parse_args()

    dash = load_dash()
    frontier = dash["status"]["frontier_tps"] if dash else args.frontier   # live ledger frontier
    prs = json.loads(gh(["pr", "list", "-R", args.repo, "--state", "open",
                         "--json", "number,headRefName,headRefOid,title,isCrossRepository"]).stdout or "[]")
    if not prs:
        print("no open PRs"); return

    # Collect PRs that actually need evaluation before starting the GPU instance.
    pending = []
    for pr in prs:
        num, branch, oid = pr["number"], pr["headRefName"], pr["headRefOid"][:7]
        ref = f"pull/{num}/head" if pr.get("isCrossRepository") else branch
        areas = areas_for_pr(args.repo, num)
        print(f"PR #{num} @ {oid}: areas={sorted(areas) or ['(none)']} ref={ref}")
        if not args.dry_run: apply_area_labels(args.repo, num, areas)
        if oid in evaluated_commits(args.repo, num):
            print(f"PR #{num} @ {oid}: already evaluated — skip eval"); continue
        pending.append((pr, num, branch, oid, ref, areas))

    if not pending:
        print("done — no merges (manual)."); return

    # Run all pending evals on the SAME instance: pass --keep so vast_eval.py never stops/destroys
    # the box mid-queue. The bot stops the instance once after ALL PRs finish (or if the instance
    # dies, subsequent PRs self-heal by provisioning a new one).
    for i, (pr, num, branch, oid, ref, areas) in enumerate(pending):
        # Re-read the frontier each iteration: a PR that passed earlier in THIS run may have
        # ratcheted it (update_dashboard writes data.json), and later PRs must be graded against
        # the new best — otherwise two PRs could both "beat" the same stale baseline.
        d = load_dash()
        cur_frontier = d["status"]["frontier_tps"] if d else frontier
        print(f"PR #{num} @ {oid}: evaluating '{ref}' (frontier={cur_frontier}) ...")
        r = subprocess.run([sys.executable, os.path.join(HERE, "vast_eval.py"),
                            "--reuse", str(current_instance(args.instance)), "--ref", ref,
                            "--frontier", str(cur_frontier), "--ceiling", str(args.ceiling),
                            "--keep"],          # keep instance alive — bot stops it after all PRs
                           cwd=ROOT, capture_output=True, text=True, timeout=14400)
        # If vast_eval self-healed to a new instance, track the new id for the next PR.
        for l in r.stdout.splitlines():
            if l.startswith("NEW_INSTANCE_ID "):
                try:
                    new_id = int(l.split()[1])
                    with open(INSTANCE_FILE, "w") as f: f.write(str(new_id))
                    print(f"  (instance updated to {new_id})")
                except Exception: pass
        line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT_JSON")), None)
        if not line:
            log = (r.stdout + r.stderr)[-1500:]
            print(f"PR #{num}: eval produced no result\n{log}")
            body = (f"<!-- sparkinfer-eval:{oid} -->\n⚠️ **sparkinfer auto-eval errored** for `{oid}` "
                    f"— re-run manually.\n\n<details><summary>log tail</summary>\n\n```\n{log}\n```\n</details>")
            res, label = None, None
        else:
            res = json.loads(line[len("RESULT_JSON "):]); label = res["label"]; body = render(res, oid)
            print(f"PR #{num}: {json.dumps(res)}")
        if args.dry_run:
            print("--- dry-run, not posting ---\n" + body); continue
        if label:
            for lab in {l for l in labels_on(args.repo, num) if l.startswith("eval:")}:
                remove_label(args.repo, num, lab)
            add_label(args.repo, num, f"eval:{label}")
        gh(["pr", "comment", str(num), "-R", args.repo, "--body", body])
        print(f"PR #{num}: posted {'eval:'+label if label else 'error'} — NOT merged.")
        if res: update_dashboard(args.repo, pr, areas, res)

    # Stop (not destroy) the instance after all PRs — disk/model cache persists for next run.
    final_iid = current_instance(args.instance)
    if final_iid:
        print(f">> stopping instance {final_iid} — model cache persists for next run")
        subprocess.run(["vastai", "stop", "instance", str(final_iid)], capture_output=True)
    print("done — no merges (manual).")

if __name__ == "__main__":
    main()
