#!/bin/bash

# Create a new tmux session or attach to an existing one with the name "llux"
SESSION_NAME="llux"

# Check if the session already exists
if ! tmux has-session -t "$SESSION_NAME"; then
  # If not, create it and run the python script inside
  tmux new-session -d -s "$SESSION_NAME" "python /Users/sij/home/workshop/llux/llux.py --debug"
else
  # If the session exists, attach to it and send the command to a new window
  tmux new-window -t "$SESSION_NAME" "python /Users/sij/home/workshop/llux/llux.py --debug"
fi

# Optionally, you can also detach from the session immediately after creating/attaching to it
tmux detach-client -s "$SESSION_NAME"


