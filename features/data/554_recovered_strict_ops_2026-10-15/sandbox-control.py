import json, socket, subprocess, sys

def check():
    refused = []
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
        except OSError:
            refused.append(int(family))
        else:
            sock.close()
            raise RuntimeError("Internet-family socket restriction absent")
    left, right = socket.socketpair()
    left.sendall(b"proof")
    assert right.recv(5) == b"proof"
    left.close()
    right.close()
    return refused

if len(sys.argv) > 1:
    print(json.dumps({"child_inet_refused": check(), "unix_ipc": True}))
else:
    print(json.dumps({"parent_inet_refused": check(), "unix_ipc": True}))
    subprocess.run([sys.executable, __file__, "child"], check=True, timeout=5)
