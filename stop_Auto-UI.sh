#!/bin/bash
pids=$(ps aux | grep "Auto-UI" | grep -v grep | awk '{print $2}')
for pid in $pids; do
    kill -9 $pid
done