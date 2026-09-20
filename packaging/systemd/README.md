# Making the memory guards start by themselves

Two separate things, and the second one matters more than the first.

## 1. The watchdog (item 450)

It only guards while something is running it. As a user unit it guards
every session on the machine, including ones that forgot to start it --
which is how 2026-09-20 went.

```sh
install -Dm755 tools/memory_watchdog.py ~/.local/bin/spacr_memory_watchdog.py
install -Dm644 packaging/systemd/spacr-memory-watchdog.service \
    ~/.config/systemd/user/spacr-memory-watchdog.service
systemctl --user daemon-reload
systemctl --user enable --now spacr-memory-watchdog
systemctl --user status spacr-memory-watchdog --no-pager
```

The script is COPIED to `~/.local/bin` rather than run from the checkout on
purpose: a unit that points into a git worktree stops working the moment
somebody moves, rebases or deletes it, and the guard is then missing
without anybody noticing. Re-run the first line after changing the script.

To watch what it does:

```sh
journalctl --user -u spacr-memory-watchdog -f
```

A process it freezes is STOPPED, not killed. `kill -CONT <pid>` puts it
back; the log line says so each time, because a frozen process looks
exactly like a hung one.

## 2. Back-pressure above every scope, which a poller cannot give you

`user@1000.service` ships with `MemoryHigh=infinity`, so nothing slows an
allocation down until the kernel starts killing. `MemoryHigh` is not a
limit that kills: above it the kernel reclaims hard and the offender is
throttled, which is exactly the "reduce" a watchdog cannot do from
outside.

This is the one guard here that needs root, because `user@.service` is a
system unit:

```sh
sudo mkdir -p /etc/systemd/system/user@.service.d
sudo tee /etc/systemd/system/user@.service.d/memoryhigh.conf <<'EOF'
[Service]
MemoryHigh=100G
EOF
sudo systemctl daemon-reload
```

It takes effect for a user session started after that, so log out and in
(or reboot). Check it with:

```sh
systemctl show user@1000.service -p MemoryHigh
```

**100G of this machine's 125G was the maintainer's choice on 2026-09-20**,
the same number the watchdog acts on. The two are deliberately the same: at
100 GB the kernel starts throttling AND the watchdog starts looking for
something to freeze, so the slow path and the decisive one begin together.

## What none of this replaces

`tools/run_capped.sh`. A cgroup with `MemoryMax` is checked by the kernel on
every allocation; everything on this page is either slower than that or
outside the process. Start python through the cap, every time, and none of
this has to work.
