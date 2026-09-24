.. _system-requirements:

System requirements
===================

Choose hardware for the largest images, number of concurrent fields and models
you plan to run. These are planning recommendations, not measured minimums or
speed guarantees. A GPU is optional for CPU-capable workflows; the acceleration
matrix below identifies where particular hardware can help. See the
:doc:`installer guide <installer_guide>` for installation and package compatibility.

General hardware recommendations
--------------------------------

.. list-table:: Suggested workstation configurations
   :header-rows: 1
   :widths: 16 28 28 28

   * - Resource
     - Low: learning and small experiments
     - Medium: routine microscopy plates
     - High: large screens and model training
   * - CPU
     - 4–8 modern cores; about 2.5–3.5 GHz sustained on x86
     - 8–16 cores; about 3–4 GHz sustained on x86
     - 16–32 cores; about 3–4 GHz sustained on x86
   * - RAM
     - 16 GB; process a few fields at a time
     - 32–64 GB
     - 128 GB or more
   * - GPU memory
     - CPU operation, or a compatible GPU with 8 GB for modest tiled inference
     - Compatible GPU with 12–16 GB dedicated VRAM; Apple silicon with 32–64 GB unified memory
     - Compatible GPU with 24–48 GB or more dedicated VRAM; Apple silicon with 64–128 GB or more unified memory for MPS-capable work
   * - Storage
     - SSD; reserve 30 GB for the environment, caches and initial models, plus project space
     - 1–2 TB NVMe SSD; reserve 50 GB for software and model caches
     - 2–4 TB NVMe scratch storage, separate archive storage; reserve 100 GB for multiple model environments and caches
   * - Typical use
     - Tutorials, annotation, classical segmentation and small CPU analyses
     - Mask, Measure, phenotype inference and moderate embeddings
     - Larger training batches, many channels, larger volumes and concurrent plate analysis

CPU clock figures are rough x86 purchasing targets, not requirements; GHz is
not comparable across architectures. Count physical cores separately from
threads, and distinguish performance from efficiency cores on hybrid CPUs.
Apple unified memory is shared by the OS, CPU and GPU and is not equivalent to
the same quantity of dedicated VRAM. Available memory, image dimensions and
batch size determine whether a workload fits. Training usually needs more
memory than inference; large 3-D images may exceed any of these examples.

Use a 64-bit OS and Python environment. The desktop installers target Windows
10/11 x86-64, Linux x86-64 and Intel/Apple-silicon macOS; their advertised OS
floor does not guarantee that a newer GPU framework supports that same OS.
Python 3.12 is the practical starting point for the broadest spaCR extras.
spaCR's declared Python range is 3.9–3.14, excluding 3.14.1, but optional
packages and individual hardware backends narrow that range. Current Apple
MPS requirements include macOS 14 or later; use the requirements for the
specific PyTorch build you install. `Apple's PyTorch requirements
<https://developer.apple.com/metal/pytorch/>`_ describe the current native
Apple-silicon route.

GPU support by chip and framework
---------------------------------

**Available** means spaCR has a dispatch route when the matching framework,
driver and model operators work. It does not mean every configuration in the
row has been tested. **Conditional** means an optional or legacy stack needs
verification with the intended module. **No** means that framework does not
provide that acceleration route; CPU execution may still be available.

.. list-table:: Compute compatibility, checked 23 September 2026
   :header-rows: 1
   :widths: 24 22 22 16 16

   * - Hardware
     - PyTorch neural workflows
     - Cellpose 4 / Cellpose-SAM
     - CuPy array operations
     - RAPIDS / cuML embeddings
   * - Apple M1 family, including Pro, Max and Ultra where offered
     - Available: Metal/MPS on compatible macOS
     - Available: MPS network; some flow operations use CPU
     - No
     - No
   * - Apple M2 family, including Pro, Max and Ultra where offered
     - Available: Metal/MPS
     - Available: MPS network; some flow operations use CPU
     - No
     - No
   * - Apple M3 family, including Pro, Max and Ultra where offered
     - Available: Metal/MPS
     - Available: MPS network; some flow operations use CPU
     - No
     - No
   * - Apple M4 family, including Pro and Max
     - Available: Metal/MPS
     - Available: MPS network; some flow operations use CPU
     - No
     - No
   * - Newer Apple silicon
     - Conditional on the installed macOS/PyTorch supporting the GPU
     - Same MPS and operator requirements
     - No
     - No
   * - Intel Mac CPU with Intel integrated graphics
     - CPU; Intel integrated graphics is not the MPS route
     - CPU
     - No
     - No
   * - Intel Mac with a compatible AMD discrete GPU
     - Conditional: legacy x86 PyTorch MPS build
     - Conditional: legacy MPS stack; spaCR can use float32 weights and CPU flows
     - No supported spaCR macOS route
     - No
   * - Intel Arc A/B discrete GPUs, supported Core Ultra Arc integrated GPUs, Data Center GPU Max
     - Available: XPU with a compatible PyTorch build and Intel driver
     - Conditional: explicit XPU device; verify Cellpose operators and build
     - No
     - No
   * - Older Intel HD/UHD/Iris integrated GPUs on Windows
     - Conditional: DirectML on DirectX 12 hardware; otherwise CPU
     - Conditional and unvalidated via DirectML; CPU is the baseline
     - No
     - No
   * - AMD Radeon RX 7900 / supported RX 9000 models, supported Radeon PRO and Instinct GPUs
     - Available: ROCm on explicitly supported GPU/OS combinations; Windows routes are version-specific
     - Available on compatible Linux ROCm; verify other OS combinations
     - Experimental ROCm upstream; not installed by spaCR's CUDA extra
     - No
   * - Other AMD integrated or discrete GPUs
     - Conditional: only if named in AMD's ROCm matrix, or DirectML on Windows; otherwise CPU
     - Conditional on the selected backend and operators
     - Experimental only for ROCm-supported hardware
     - No
   * - NVIDIA GeForce RTX 20/30/40/50, compatible RTX workstation and data-center GPUs
     - Available: CUDA on Linux/Windows with a matching wheel and driver
     - Available: CUDA; batch/image size must fit VRAM
     - Available: matching CUDA CuPy package
     - Available on supported Linux/WSL2; release-specific compute capability and CUDA requirements
   * - Intel/AMD CPUs without a supported GPU
     - CPU, not GPU acceleration
     - CPU
     - No; use NumPy/SciPy fallbacks
     - No; use CPU embedding implementations

The processor brand does not determine CUDA availability: an AMD Ryzen or Intel
Core host can both use an NVIDIA GPU. Likewise, an NPU or Apple Neural Engine
is not a PyTorch GPU device in spaCR. Detection of such hardware is not a claim
that Mask or Classify uses it.

The Apple rows share the MPS route; increasing the chip generation does not
add CUDA or CuPy support. spaCR's device resolver checks the actual runtime
and selects float32 Cellpose weights when bfloat16 is unavailable. The current
Apple guidance targets Apple silicon; Intel Mac/AMD support is a legacy path
with older compatible PyTorch wheels, not a recommendation to buy an Intel Mac.
`PyTorch MPS documentation <https://docs.pytorch.org/docs/stable/notes/mps.html>`_
and `Cellpose installation documentation
<https://cellpose.readthedocs.io/en/latest/installation.html>`_ explain their
backend requirements.

Intel's supported devices and OS combinations differ by generation. Check the
`PyTorch XPU hardware table
<https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>`_ for the exact
Arc/Core Ultra model; an Intel logo alone is insufficient. DirectML is a
separate optional Windows backend with its own package and operator limits;
Microsoft labels the PyTorch integration public preview. See
`PyTorch with DirectML <https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows>`_.

For AMD, select the exact GPU, OS and PyTorch release in the
`ROCm compatibility matrix
<https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html>`_.
Support for one RX 7000 or RX 9000 model does not imply support for every card
in that series. spaCR recognizes ROCm through PyTorch's CUDA-compatible API
and reports it separately from NVIDIA CUDA.

For NVIDIA, check the `GPU compute-capability table
<https://developer.nvidia.com/cuda/gpus>`_ and use the matching build from the
`PyTorch installation selector <https://pytorch.org/get-started/locally/>`_.
New GPU generations may require newer CUDA wheels; an older installation can
fail despite the card being CUDA-capable. CuPy and RAPIDS are independent of
PyTorch: installing a CUDA-enabled torch wheel alone does not install them.
`CuPy's installation guide <https://docs.cupy.dev/en/stable/install.html>`_
describes CUDA packages and the experimental AMD path. The
`RAPIDS platform requirements <https://docs.rapids.ai/install/>`_ impose their
own GPU architecture, CUDA and OS restrictions; native Windows is not the
standard RAPIDS route. spaCR's ``rapids`` extra currently requests CUDA 12
packages on Python 3.11–3.12. Check the selected RAPIDS release rather than
assuming every CUDA GPU works.

Which spaCR operations use these backends?
------------------------------------------

.. list-table:: Runtime paths
   :header-rows: 1
   :widths: 28 36 36

   * - Operation
     - Acceleration
     - Practical limit or fallback
   * - Mask / Make Masks / live Cellpose preview
     - PyTorch through spaCR's device resolver and Cellpose 4
     - Network, flow reconstruction and image preprocessing can use different devices; CPU remains usable
   * - Phenotype classification, training and neural inference
     - PyTorch CUDA, ROCm, MPS, XPU or optional DirectML, as resolved
     - Model-specific operators, precision and training memory may narrow support
   * - Optional isolated segmentation backends
     - Their own environment's torch build; automatic worker selection currently checks CUDA/ROCm, then MPS, then CPU
     - Host detection does not install a matching worker stack; do not assume automatic XPU/DirectML worker dispatch
   * - OPS registration, peak finding, unmixing and matching primitives
     - PyTorch where available, optional CuPy operations, then CPU implementations
     - ``ops_gpu`` and operation-specific backend options govern use; not every registration step is a GPU operation
   * - Optional accelerated dimensionality reduction
     - RAPIDS/cuML with CUDA and CuPy
     - Explicit opt-in; CPU fallback when unavailable or when deterministic behavior requires it
   * - Measurement tables, image I/O and many classical statistics
     - Predominantly CPU, RAM and disk throughput
     - Adding VRAM does not replace host RAM or speed up every module

The matrix is based on spaCR's ``accelerator``, ``object``, ``ops_accel``,
``gpu_reduce`` and isolated-backend dispatch code as well as the linked
upstream documentation. Framework capability is broader than a tested spaCR
workflow. Verify the device reported by the intended module with a small
representative field before scheduling a whole plate.

Servers and shared compute
--------------------------

.. list-table:: Suggested server planning ranges
   :header-rows: 1
   :widths: 22 39 39

   * - Resource
     - One active analysis job
     - Several independent jobs / larger training
   * - CPU
     - 16–32 physical cores; approximately 2.5–3.5 GHz sustained x86 clock
     - 32–64 or more physical cores; approximately 2.5–3.5 GHz sustained, with adequate memory bandwidth
   * - RAM
     - 128 GB; more for large 3-D fields or large measurement tables
     - 256–512 GB or more, apportioned per job
   * - GPU
     - One compatible GPU with 24–48 GB VRAM for a broadly useful training/inference node
     - Multiple compatible GPUs, typically 48–80 GB or more each for larger jobs; assign jobs explicitly
   * - Local disk
     - 2–4 TB NVMe scratch, plus 50–100 GB for environments and model caches
     - 4–8 TB or more NVMe scratch; size for all simultaneously active projects
   * - Archive / network storage
     - Capacity for raw data, retained results and an independent backup
     - Shared storage sized for aggregate throughput; 10 GbE or faster is a useful planning target when moving large plates

These ranges are recommendations, not a guarantee that a particular volume or
model will fit. Several GPUs do not automatically combine their VRAM into a
single larger GPU. Independent job scheduling is distinct from distributed
model training; use a module's documented multi-GPU capability when available.
Limit worker counts and library threads to the scheduler allocation and leave
RAM for the OS, caches and the parent process. Begin with one GPU worker per
allocated GPU, then measure utilization before increasing concurrency.

Run headless jobs through ``spacr-run`` without launching a desktop or display
server; ``spacr-run --list`` shows the available pipelines. Install the correct
GPU framework in the environment actually running the job. Containers still
require a compatible host driver and access to the assigned GPU. Use local
scratch for active array processing and stage completed artifacts to shared
storage, with a distinct writable project/output directory for each job.

Disk and memory budgeting
-------------------------

Budget from uncompressed data, not only the compressed microscope files. An
array needs approximately ``Y × X × Z × channels × bytes_per_value`` bytes,
multiplied by the number of fields held at once. A 2048 × 2048, four-channel
uint16 field is 32 MiB before masks or working copies; float32 doubles that
image allocation. Normalized copies, intermediate arrays, masks, crops,
training tensors and tables add separate allocations, and model activations
can dominate GPU memory.

As an initial disk reservation, allow 3–5 times the uncompressed input size
for an active project, in addition to software/model space and backups. This
is a planning allowance, not an upper bound: extensive object crops, many
retained checkpoints, volumes or repeated runs can require substantially
more. Process a representative subset, measure the actual output multiplier,
and reserve free space before expanding to the whole screen. Keep backups
outside the working-space allowance. Each optional model environment and its
weights can add gigabytes independently of the main spaCR installation.
