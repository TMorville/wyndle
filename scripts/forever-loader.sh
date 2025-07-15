#!/bin/bash

# forever-loader.sh
# Sets up Wyndle continuous loader as a systemd user service

echo "Setting up Wyndle continuous loader as systemd user service..."

# Create systemd user service file
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/wyndle-loader.service << 'EOF'
[Unit]
Description=Wyndle Continuous Loader
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/env uv run wyndle-loader
WorkingDirectory=%h/Projects/wyndle
Restart=always
RestartSec=10
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

echo "Created systemd service file at ~/.config/systemd/user/wyndle-loader.service"

# Reload systemd user daemon
systemctl --user daemon-reload

# Enable and start the service
systemctl --user enable --now wyndle-loader

echo "✅ Wyndle continuous loader is now running as a systemd service!"
echo ""
echo "Useful commands:"
echo "  systemctl --user status wyndle-loader     # Check status"
echo "  journalctl --user -u wyndle-loader -f     # View live logs"
echo "  systemctl --user stop wyndle-loader       # Stop service"
echo "  systemctl --user restart wyndle-loader    # Restart service"
echo "  systemctl --user disable wyndle-loader    # Disable auto-start" 