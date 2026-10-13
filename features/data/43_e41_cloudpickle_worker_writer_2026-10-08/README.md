# e41 worker-writer serializer compatibility

Hosted Coverage5 installed joblib 1.6.0 and cloudpickle 3.1.2. Four original
worker-writer cases failed before their scientific and queue assertions because
the code imported the removed `joblib.externals.cloudpickle` path. The repair
uses the bundled codec on older joblib and standalone cloudpickle on the
current layout, in both caller and spawned worker. It changes no worker count,
retry order, queued write, or dependency declaration.

With the same two package versions in a scratch-only overlay, the entire
owning file passed 43 cases and the adjacent real two-worker closure and
final-retry cases passed. The older local joblib 1.4.2 vendor path passed those
two adjacent cases too. The frozen hosted log includes other Coverage5 tests;
this archive only claims the four listed worker-writer failures.

Run `python verify.py --git --current` on the source-bound revision.
