Ordinary original-fault context proof, 2026-10-07

882f2dcd1a adds ordinary-only bounded fault-thread evidence before the all-thread
trace: siginfo/current full bt24, then at most24 frames to the first saved signal
context, original registers and16 instructions. Errors are printed without
suppressing existing traces. Print limits32 elements/depth4 bound local data.
The protected serial command,90s/4MiB limits, all-thread bt24 and test/application
semantics remain unchanged. No raw core is archived or uploaded.

Five fake-frame selection/error/boundary cases plus two extraction cleanup
cases pass7; the independent original serial-command guard passes1.
One tiny controlled ctypes null-pointer fault reraised by CPython faulthandler
proves the actual problem: siginfo is SI_TKILL(-6), current PC is pthread_kill,
but original frame5 is strlen with restored rdi=0 and a vpcmpeqb (%rdi) fault.
Gdb returns0. The executed harness returns1 solely because its last assertion
matches the word unsupported in the logged script source. Existing output
was checked by actual diagnostic-line prefixes; no second core was generated.
The failed harness script is preserved. Raw core cleanup completed.

Portable future reproduction from repository root (requires gdb):
  tools/run_capped.sh 4G python features/data/47_ordinary_fault_context_cpu_2026-10-07/reproduce.py
It uses a fresh private scratch directory and corrects only the final harness
prefix assertion. This reproduction was not rerun. Source/test hashes bind
the verified output to the committed implementation; the executed gitHEAD
preceded that commit with the final helper bytes already on disk.
This is diagnostic acceptance only. Hosted Qt crash causation remains OPEN.
