# run_pipeline.py
# 顺序执行一系列 Python 脚本（带日志/超时/重试），支持 CLI 与 notebook 直接调用。

import argparse, json, os, sys, time, subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

MANIFEST_DEFAULT = "pipeline.json"

def load_manifest(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"清单不存在：{p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "steps" not in data or not isinstance(data["steps"], list) or not data["steps"]:
        raise ValueError("清单文件需包含非空的 steps 列表。")
    return data

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def run_step(step: Dict[str, Any],
             default_python: str,
             log_dir: Path,
             global_timeout: Optional[int] = None) -> Dict[str, Any]:
    name = step.get("name") or step.get("script") or step.get("module") or "unnamed-step"
    retries = int(step.get("retries", 0))
    timeout = step.get("timeout", None)
    timeout = int(timeout) if timeout not in (None, "", False) else None
    if timeout is None and global_timeout:
        timeout = int(global_timeout)
    py_exe = step.get("python") or default_python or sys.executable

    if step.get("module"):
        cmd = [py_exe, "-m", str(step["module"])]
    else:
        if "script" not in step:
            raise ValueError(f"步骤缺少 script 或 module：{step}")
        cmd = [py_exe, str(step["script"])]

    if step.get("args"):
        if not isinstance(step["args"], list):
            raise ValueError(f"'args' 应为列表：{step}")
        cmd.extend([str(a) for a in step["args"]])

    cwd = step.get("cwd", None)
    env = os.environ.copy()
    if step.get("env"):
        for k, v in step["env"].items():
            env[str(k)] = str(v)

    safe_name = "".join(c if c.isalnum() or c in "-._" else "_" for c in name)[:80]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{ts}__{safe_name}.log"

    attempt = 0
    start_time = time.time()
    last_err = None
    while attempt <= retries:
        attempt += 1
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(f"$ {' '.join(cmd)}\n")
            lf.write(f"# cwd={cwd or os.getcwd()} timeout={timeout or 'none'} attempt={attempt}/{retries+1}\n\n")
            lf.flush()
            print(f"▶ [{name}] 开始（第 {attempt}/{retries+1} 次）…")
            t0 = time.time()
            try:
                res = subprocess.run(
                    cmd, cwd=cwd, env=env, text=True,
                    capture_output=True, timeout=timeout
                )
                duration = time.time() - t0
                if res.stdout:
                    lf.write(res.stdout)
                if res.stderr:
                    lf.write("\n[stderr]\n"); lf.write(res.stderr)
                lf.write(f"\n# returncode={res.returncode} duration={duration:.2f}s\n"); lf.flush()

                tail = (res.stdout or "")[-1000:]
                if tail.strip():
                    print(tail.strip())
                if res.returncode == 0:
                    print(f"✅ [{name}] 成功（{duration:.2f}s）日志：{log_file}")
                    return {"name": name, "returncode": 0, "duration_sec": duration, "log_file": str(log_file)}
                else:
                    print(f"❌ [{name}] 退出码 {res.returncode}（{duration:.2f}s），查看日志：{log_file}")
                    last_err = res
            except subprocess.TimeoutExpired as e:
                duration = time.time() - t0
                msg = f"[TIMEOUT] 超时（{timeout}s）。"
                lf.write(f"\n{msg}\n"); print(f"⏰ [{name}] {msg} 日志：{log_file}")
                last_err = e
            except Exception as e:
                duration = time.time() - t0
                msg = f"[ERROR] {e}"
                lf.write(f"\n{msg}\n"); print(f"💥 [{name}] {msg} 日志：{log_file}")
                last_err = e

        if attempt <= retries:
            print(f"↻ [{name}] 准备重试（{attempt}/{retries} 已用）…")

    total_dur = time.time() - start_time
    return {"name": name, "returncode": 1, "duration_sec": total_dur, "log_file": str(log_file), "error": str(last_err)}

def _run_core(manifest_path: str,
              default_python: Optional[str],
              log_dir: str,
              continue_on_fail: bool,
              timeout: Optional[int],
              dry_run: bool) -> int:
    manifest = load_manifest(manifest_path)
    steps: List[Dict[str, Any]] = manifest["steps"]
    default_python = manifest.get("default_python", default_python or sys.executable)
    stop_on_fail = manifest.get("stop_on_fail", True) and (not continue_on_fail)
    global_timeout = timeout or manifest.get("default_timeout", None)

    log_root = Path(log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dir(log_root)

    if dry_run:
        print("将按以下顺序执行：")
        for i, s in enumerate(steps, 1):
            nm = s.get("name") or s.get("script") or s.get("module") or f"step-{i}"
            if s.get("module"):
                base = f"{default_python} -m {s['module']}"
            else:
                base = f"{default_python} {s.get('script','<module>')}"
            extra = " " + " ".join(map(str, s.get("args", []))) if s.get("args") else ""
            print(f"{i:02d}. {nm}: {base}{extra}  (cwd={s.get('cwd','.')})")
        return 0

    results = []; failures = 0
    print(f"共 {len(steps)} 个步骤；日志目录：{log_root}")
    for idx, step in enumerate(steps, 1):
        name = step.get("name") or step.get("script") or step.get("module") or f"step-{idx}"
        print(f"\n=== [{idx:02d}/{len(steps)}] {name} ===")
        res = run_step(step, default_python=default_python, log_dir=log_root, global_timeout=global_timeout)
        results.append(res)
        if res.get("returncode", 1) != 0:
            failures += 1
            if stop_on_fail:
                print("\n⛔ 检测到失败，配置为遇错停止。结束执行。")
                break

    print("\n===== 执行总结 =====")
    for r in results:
        status = "OK" if r.get("returncode", 1) == 0 else "FAIL"
        print(f"[{status}] {r['name']}  ({r['duration_sec']:.2f}s)  log: {r['log_file']}")
    if failures:
        print(f"\n结果：{len(steps)-failures} 成功，{failures} 失败。退出码 1")
        return 1
    else:
        print(f"\n结果：全部 {len(steps)} 成功。退出码 0")
        return 0

# —— 提供给 notebook 直接调用的入口 —— #
def run(manifest_path: str = MANIFEST_DEFAULT,
        python: Optional[str] = None,
        log_dir: str = "logs/pipeline",
        continue_on_fail: bool = False,
        timeout: Optional[int] = None,
        dry_run: bool = False) -> int:
    """在 Python/Notebook 中直接调用：run('pipeline.json')"""
    if manifest_path is None or not Path(manifest_path).exists():
        raise FileNotFoundError(f"未找到清单：{manifest_path}")
    return _run_core(manifest_path, python, log_dir, continue_on_fail, timeout, dry_run)

def main():
    ap = argparse.ArgumentParser(description="顺序执行一系列 Python 脚本（带日志/超时/重试）。")
    ap.add_argument("--manifest", default=None,
                    help=f"JSON 清单路径；若省略且存在 {MANIFEST_DEFAULT} 则使用它")
    ap.add_argument("--python", default=sys.executable, help="默认 Python 解释器（可被步骤级覆盖）")
    ap.add_argument("--log-dir", default="logs/pipeline", help="日志目录")
    ap.add_argument("--continue-on-fail", action="store_true", help="遇失败继续执行后续步骤")
    ap.add_argument("--timeout", type=int, default=None, help="为所有步骤设置默认超时（秒），被步骤级 timeout 覆盖")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不实际运行")
    args = ap.parse_args()

    manifest_path = args.manifest
    if manifest_path is None:
        # 尝试默认文件或环境变量
        env_path = os.environ.get("PIPELINE_MANIFEST")
        if env_path and Path(env_path).exists():
            manifest_path = env_path
            print(f"使用环境变量 PIPELINE_MANIFEST: {manifest_path}")
        elif Path(MANIFEST_DEFAULT).exists():
            manifest_path = MANIFEST_DEFAULT
            print(f"使用默认清单: {manifest_path}")
        else:
            print(f"ERROR: 需要 --manifest；且当前目录未发现 {MANIFEST_DEFAULT}")
            sys.exit(2)

    rc = _run_core(manifest_path, args.python, args.log_dir, args.continue_on_fail, args.timeout, args.dry_run)
    sys.exit(rc)

if __name__ == "__main__":
    main()
