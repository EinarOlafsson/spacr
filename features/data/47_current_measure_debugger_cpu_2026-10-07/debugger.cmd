set pagination off
set confirm off
handle SIGSEGV stop print nopass
run
python
import gdb
inferior = gdb.selected_inferior()
if inferior.threads():
    print("SPACR_GDB_INFERIOR_STOPPED_WITH_LIVE_THREADS")
    gdb.execute("thread apply all bt 24")
    gdb.execute("info registers")
    gdb.execute("x/16i $pc-16")
else:
    print("SPACR_GDB_INFERIOR_EXITED")
end
