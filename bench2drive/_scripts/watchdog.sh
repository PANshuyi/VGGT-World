#!/bin/bash
# Watchdog script for Bench2Drive download
# Auto-restarts the download if it dies before completion
# Run: nohup bash watchdog.sh >> Dataset/Bench2Drive/_logs/watchdog.log 2>&1 &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$(dirname "$BENCH_DIR")")"
DOWNLOAD_SCRIPT="$BENCH_DIR/_scripts/robust_download.py"
LOG_DIR="$BENCH_DIR/_logs"
STATE_FILE="$LOG_DIR/download_state.txt"
VENV="$PROJECT_DIR/.venv-mapanything-convert/bin/activate"

MAX_RESTARTS=10
RESTART_COUNT=0
CHECK_INTERVAL=60  # Check every 60 seconds

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] $*"
}

log "Watchdog started. Monitoring Bench2Drive download..."
log "Project dir: $PROJECT_DIR"
log "Download script: $DOWNLOAD_SCRIPT"

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    # Check if download is already running
    RUNNING=$(ps aux | grep "robust_download.py" | grep -v grep | grep -v watchdog | wc -l)
    
    if [ "$RUNNING" -eq 0 ]; then
        # Check if download is complete
        if [ -f "$STATE_FILE" ]; then
            COMPLETED=$(grep "^completed=" "$STATE_FILE" 2>/dev/null | cut -d= -f2)
            TOTAL=$(grep "^total=" "$STATE_FILE" 2>/dev/null | cut -d= -f2)
            FAILED=$(grep "^failed=" "$STATE_FILE" 2>/dev/null | cut -d= -f2)
            
            if [ "$COMPLETED" = "$TOTAL" ] && [ "$COMPLETED" != "" ] && [ "$FAILED" = "0" ]; then
                log "✅ Download appears complete! ($COMPLETED/$TOTAL files)"
                log "Running final validation..."
                cd "$PROJECT_DIR"
                source "$VENV"
                python3 -c "
from huggingface_hub import list_repo_files
import os
files = list_repo_files('rethinklab/Bench2Drive', repo_type='dataset')
local = 'Dataset/Bench2Drive'
missing = [f for f in files if not os.path.exists(os.path.join(local, f))]
incomplete = [f for f in files if os.path.exists(os.path.join(local, f)) and os.path.getsize(os.path.join(local, f)) < 1024]
print(f'Missing: {len(missing)}, Incomplete: {len(incomplete)}')
if not missing and not incomplete:
    print('ALL FILES DOWNLOADED SUCCESSFULLY!')
    exit(0)
else:
    exit(1)
"
                if [ $? -eq 0 ]; then
                    log "🎉 All files validated! Exiting watchdog."
                    exit 0
                fi
            fi
        fi
        
        # Restart download
        RESTART_COUNT=$((RESTART_COUNT + 1))
        log "⚠️  Download process not running. Restarting (attempt $RESTART_COUNT/$MAX_RESTARTS)..."
        cd "$PROJECT_DIR"
        source "$VENV"
        nohup python3 -u "$DOWNLOAD_SCRIPT" >> "$LOG_DIR/download_robust_stdout.log" 2>&1 &
        PID=$!
        log "Started new download process with PID: $PID"
    else
        # Download is running - show quick status
        if [ -f "$LOG_DIR/download_robust.log" ]; then
            LAST_LINE=$(tail -1 "$LOG_DIR/download_robust.log" 2>/dev/null)
            log "Running: $LAST_LINE"
        fi
    fi
    
    sleep $CHECK_INTERVAL
done

log "❌ Max restarts ($MAX_RESTARTS) reached. Manual intervention needed."
