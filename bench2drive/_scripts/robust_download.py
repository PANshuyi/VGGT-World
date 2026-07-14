#!/usr/bin/env python3
"""
Robust Bench2Drive downloader with:
- Auto-resume (curl -C -)
- Retry on failure (3 attempts)
- Incomplete file detection & re-download
- Progress logging
- 8 parallel workers
"""

import subprocess
import os
import sys
import time
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

BASE_URL = "https://huggingface.co/datasets/rethinklab/Bench2Drive/resolve/main"
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Dataset/Bench2Drive
LOG_FILE = os.path.join(LOCAL_DIR, "_logs", "download_robust.log")
FAILED_FILE = os.path.join(LOCAL_DIR, "_logs", "download_failed.txt")
STATE_FILE = os.path.join(LOCAL_DIR, "_logs", "download_state.txt")
NUM_WORKERS = 8
MAX_RETRIES = 3
MIN_FILE_SIZE = 1024  # files smaller than this are considered incomplete

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Track state for graceful shutdown
shutdown_flag = False
active_downloads = {}

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def signal_handler(sig, frame):
    global shutdown_flag
    log(f"Received signal {sig}, shutting down gracefully...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def get_file_list():
    """Get list of all files in the HF repo."""
    from huggingface_hub import list_repo_files
    log("Fetching file list from HuggingFace...")
    files = list_repo_files('rethinklab/Bench2Drive', repo_type='dataset')
    log(f"Total files on HF: {len(files)}")
    return files

def check_local_files(all_files):
    """Check which files are missing or incomplete."""
    missing = []
    incomplete = []
    downloaded = []
    
    for f in all_files:
        local_path = os.path.join(LOCAL_DIR, f)
        if not os.path.exists(local_path):
            missing.append(f)
        elif os.path.getsize(local_path) < MIN_FILE_SIZE:
            incomplete.append(f)
            # Remove incomplete file so curl -C doesn't try to resume a tiny file
            os.remove(local_path)
            missing.append(f)
        else:
            downloaded.append(f)
    
    return downloaded, missing, incomplete

def download_file(fname):
    """Download a single file with retries. Returns (fname, success, error_msg)."""
    global shutdown_flag
    
    url = f"{BASE_URL}/{fname}"
    out = os.path.join(LOCAL_DIR, fname)
    
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(out), exist_ok=True)
    
    for attempt in range(1, MAX_RETRIES + 1):
        if shutdown_flag:
            return (fname, False, "Shutdown requested")
        
        try:
            # Use curl with resume support
            cmd = [
                "curl", "-sS", "-L", "-C", "-",
                "--connect-timeout", "30",
                "--max-time", "3600",
                "--retry", "2",
                "--retry-delay", "10",
                "-o", out, url
            ]
            
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3700)
            
            if r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) >= MIN_FILE_SIZE:
                return (fname, True, None)
            else:
                # If file exists but is too small, remove it before retry
                if os.path.exists(out) and os.path.getsize(out) < MIN_FILE_SIZE:
                    os.remove(out)
                
                if attempt < MAX_RETRIES:
                    wait = attempt * 30
                    log(f"  Retry {attempt}/{MAX_RETRIES} for {fname} (waiting {wait}s)...")
                    time.sleep(wait)
                    
        except subprocess.TimeoutExpired:
            log(f"  Timeout on attempt {attempt} for {fname}")
            if attempt < MAX_RETRIES:
                time.sleep(30)
        except Exception as e:
            log(f"  Error on attempt {attempt} for {fname}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(30)
    
    return (fname, False, f"Failed after {MAX_RETRIES} attempts")

def main():
    log("=" * 60)
    log("Robust Bench2Drive Downloader Started")
    log(f"Workers: {NUM_WORKERS}, Max retries: {MAX_RETRIES}")
    log("=" * 60)
    
    # Get file list and check local state
    all_files = get_file_list()
    downloaded, missing, incomplete = check_local_files(all_files)
    
    log(f"Already downloaded: {len(downloaded)}")
    log(f"Incomplete (re-downloading): {len(incomplete)}")
    log(f"Missing (need download): {len(missing)}")
    
    if not missing:
        log("✅ All files downloaded! Nothing to do.")
        
        # Final validation
        log("Running final validation...")
        _, still_missing, still_incomplete = check_local_files(all_files)
        if still_missing:
            log(f"⚠️  Still missing: {len(still_missing)}")
        if still_incomplete:
            log(f"⚠️  Still incomplete: {len(still_incomplete)}")
        if not still_missing and not still_incomplete:
            log("✅ Validation passed! All 1002 files are complete.")
        return
    
    total = len(missing)
    completed = 0
    failed = []
    
    log(f"Starting download of {total} files...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(download_file, f): f for f in missing}
        
        for future in as_completed(futures):
            if shutdown_flag:
                log("Shutdown flag set, cancelling remaining downloads...")
                for f in futures:
                    futures[f].cancel()
                break
            
            fname, success, error = future.result()
            completed += 1
            
            if success:
                pct = completed / total * 100
                elapsed = time.time() - start_time
                rate = completed / elapsed * 60 if elapsed > 0 else 0
                eta_min = (total - completed) / rate if rate > 0 else 0
                log(f"[{completed}/{total}] ({pct:.1f}%) OK: {fname}  |  {rate:.1f} files/min  |  ETA: {eta_min:.0f}min")
            else:
                failed.append(fname)
                log(f"[{completed}/{total}] ({completed/total*100:.1f}%) ❌ FAIL: {fname}  -  {error}")
            
            # Save state periodically
            if completed % 10 == 0:
                with open(STATE_FILE, "w") as sf:
                    sf.write(f"completed={completed}\ntotal={total}\nfailed={len(failed)}\n")
    
    # Save failed files for re-download
    if failed:
        with open(FAILED_FILE, "w") as ff:
            for f in failed:
                ff.write(f + "\n")
        log(f"⚠️  {len(failed)} files failed. Saved to {FAILED_FILE}")
    
    elapsed_total = time.time() - start_time
    log(f"=== Download session finished in {elapsed_total/60:.1f} min ===")
    log(f"Completed: {completed - len(failed)}, Failed: {len(failed)}")
    
    # Final summary
    _, still_missing, still_incomplete = check_local_files(all_files)
    total_local = len([f for f in os.listdir(LOCAL_DIR) if not f.startswith('_')])
    log(f"Final state: {total_local} files in directory")
    if still_missing:
        log(f"Still missing: {len(still_missing)} files")
    if still_incomplete:
        log(f"Still incomplete: {len(still_incomplete)} files")
    
    if failed:
        log(f"Run this script again to retry {len(failed)} failed files.")
    elif not still_missing:
        log("✅ ALL FILES DOWNLOADED SUCCESSFULLY! 🎉")

if __name__ == "__main__":
    main()
