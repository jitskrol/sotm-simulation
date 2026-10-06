#!/bin/bash
# Hebbian Simulation launcher for macOS. Double-click in Finder to start.
#
# The simulation needs Python with GTK4. On a Mac these come from Homebrew
# (https://brew.sh). On first run this launcher offers to install the required
# packages, then starts the simulation. Later runs start it straight away.
#
# The first time, macOS may refuse to open it because it was not downloaded
# from the App Store: right-click the file, choose "Open", then "Open" again.

cd "$(dirname "$0")" || exit 1

PACKAGES="gtk4 pygobject3 py3cairo numpy adwaita-icon-theme"

has_everything() {
    "$1" -c "import gi; gi.require_version('Gtk', '4.0'); from gi.repository import Gtk; import numpy, cairo" \
        >/dev/null 2>&1
}

find_python() {
    for p in /opt/homebrew/bin/python3 /usr/local/bin/python3 \
             /opt/homebrew/bin/python3.* /usr/local/bin/python3.* "$(command -v python3)"; do
        if [ -x "$p" ] && has_everything "$p"; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

close_prompt() {
    echo
    read -r -p "Press Enter to close this window. "
}

PY=$(find_python)
if [ -z "$PY" ]; then
    BREW=$(command -v brew)
    [ -z "$BREW" ] && [ -x /opt/homebrew/bin/brew ] && BREW=/opt/homebrew/bin/brew
    [ -z "$BREW" ] && [ -x /usr/local/bin/brew ] && BREW=/usr/local/bin/brew
    if [ -z "$BREW" ]; then
        echo "The simulation needs Homebrew, which provides Python and GTK4 for macOS."
        echo "Install it by following the instructions on https://brew.sh,"
        echo "then double-click this launcher again."
        close_prompt
        exit 1
    fi
    echo "Python, GTK4 and numpy still need to be installed with Homebrew"
    echo "(only needed once): $PACKAGES"
    read -r -p "Install them now? [y/N] " answer
    case "$answer" in
        [yY]*) "$BREW" install $PACKAGES || { echo "The installation failed."; close_prompt; exit 1; } ;;
        *) exit 1 ;;
    esac
    PY=$(find_python)
    if [ -z "$PY" ]; then
        echo "The packages were installed, but no Python with GTK4 was found."
        close_prompt
        exit 1
    fi
fi

echo "Starting the simulation with $PY …"
"$PY" hebbian_sim.py || close_prompt
