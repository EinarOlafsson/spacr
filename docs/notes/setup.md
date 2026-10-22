# Notes from `setup.py`

Prose lifted out of `setup.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Current dependency decisions (2026-10-21)

This section supersedes the historical bounds and optional-package decisions below.
All 127 unique distributions declared in `setup.py` were checked against official
PyPI release JSON, including every extra and interpreter marker. The linked
release metadata supplies `Requires-Python`, `Requires-Dist` and wheel filenames.
“3.10 latest” means interpreter-compatible release metadata, not a promise that
all such releases resolve together. Dependencies shared by extras are listed once;
the roles column accounts for every named extra. Named extras for promoted
features remain aliases. Development/build/test tools remain in `dev`.

The policy remains newest compatible versions, with the existing lower bounds
preserved as API/minimum-profile promises. A next-major or next-minor API-family
guard is retained when it admits the current stable release; it is not evidence
that an unpublished release fails. Every guard blocking a published stable
release was widened and exercised, except the concrete exclusions below.
Unbounded packages remain unbounded. No lockfile or duplicate transitive caps were
added. The existing Python 3.9 metadata/minimum job is untouched; qualification
in this audit targets Python 3.10 and newer.

Clean Linux CPU installations exercised CPython 3.10.22, 3.12.14 and 3.14.7.
The exact latest-all trial exposed RAPIDS/Numba, DINOCell pins, Lifelines/pandas,
and OMERO/Ice build conflicts. A second joint solve let GPyTorch select mpmath
1.3.0 and aiobotocore select botocore without imposing project caps. Thirty-eight
compatible runtime feature distributions are now default dependencies on their
qualified interpreters. Their default set resolves together with NumPy 2,
current Cellpose and CPU Torch. Python 3.10 resolves older releases using upstream
`Requires-Python` and joint constraints, including Ultrack 0.6.3/VisPy 0.16.2/Zarr
2.18.3; this is preferable to dropping Python 3.10.

Native Windows/macOS, microscope/OMERO/cloud credentials, speech models and GPU
execution were not exercised. Existing CI owns those native platform runs.
CPU installers still select CPU Torch before installing the package; newly
promoted features add no CUDA backend. CUDA-feature, DirectML and incompatible
Intel-extension packages remain explicit extras. The already-core full XGBoost
wheel carries NCCL on Linux; retaining it preserves existing GPU algorithms.
This audit executes only CPU inference and never initializes CUDA.
Full OpenCV and its dependents remain optional because [OpenCV documents a
single cv2 distribution per environment](https://pypi.org/project/opencv-python-headless/).
The [Intel extension retirement guidance](https://intel.github.io/intel-extension-for-pytorch/latest/)
recommends standard Torch. Qt's precise 6.12.0 exclusion preserves the earlier
Python 3.10 fatal painter reproduction (commit `621bf7433fb3`), despite small
newer-interpreter tests passing that release.

| Distribution / official metadata | Latest stable | Requires-Python | 3.10 latest | Declaration roles | Bounds decision |
| --- | --- | --- | --- | --- | --- |
| [adjustText](https://pypi.org/pypi/adjustText/json) | 1.4.0 | `unspecified` | 1.4.0 | default | `<2.0,>=1.2.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [adlfs](https://pypi.org/pypi/adlfs/json) | 2026.8.0 | `>=3.10` | 2026.8.0 | default, cloud | `>=2023.1`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [aiohttp](https://pypi.org/pypi/aiohttp/json) | 3.14.5 | `>=3.10` | 3.14.5 | default, cloud | `<4,>=3.8`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [anndata](https://pypi.org/pypi/anndata/json) | 0.13.4 | `>=3.12` | 0.11.4 | default, all, anndata | `<0.14,>=0.10`. Promote; raise minor-family cap to <0.14 after 0.13 export/HDF5 round-trips with pandas 3. |
| [biopython](https://pypi.org/pypi/biopython/json) | 1.88 | `>=3.10` | 1.88 | default | `<2.0,>=1.80`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [botocore](https://pypi.org/pypi/botocore/json) | 1.43.116 | `>=3.10` | 1.43.116 | default, cloud | `<2,>=1.29`. Promote; keep broad <2; aiobotocore/s3fs select compatible botocore. No duplicate spaCR pin. |
| [bottleneck](https://pypi.org/pypi/bottleneck/json) | 1.6.0 | `>=3.10` | 1.6.0 | default | `<2.0,>=1.3.6`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [btrack](https://pypi.org/pypi/btrack/json) | 0.7.0 | `>=3.9` | 0.7.0 | default, all, btrack | `<1.0,>=0.7.0`. Promote; on Mac Python 3.14 require Darwin >=24 (macOS 15): transitive CVXOPT has no older-system cp314 wheel. Other default platforms/interpreters unchanged. |
| [captum](https://pypi.org/pypi/captum/json) | 0.9.0 | `>=3.10` | 0.9.0 | default | `<1.0,>=0.7.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [catboost](https://pypi.org/pypi/catboost/json) | 1.2.10 | `unspecified` | 1.2.10 | default, all, boosting | `<2.0,>=1.2`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [cellpose](https://pypi.org/pypi/cellpose/json) | 4.2.1.1 | `unspecified` | 4.2.1.1 | default | `<5.0,>=4.0.7`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [cloudpickle](https://pypi.org/pypi/cloudpickle/json) | 3.1.2 | `>=3.8` | 3.1.2 | default | `<4,>=3.1.2`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [cucim-cu12](https://pypi.org/pypi/cucim-cu12/json) | 26.10.0 | `>=3.11` | 26.2.1 | gpu | `>=24.0`. Keep CUDA extra: CUDA 12 native stack; RAPIDS 26.10 also requires Numba <0.65, conflicting with newest 0.68. |
| [cuml-cu12](https://pypi.org/pypi/cuml-cu12/json) | 26.10.0 | `>=3.11` | 26.2.0 | rapids | `>=25.2`. Keep CUDA extra: CUDA 12 native stack; RAPIDS 26.10 requires Numba <0.65, conflicting with newest 0.68. |
| [cupy-cuda12x](https://pypi.org/pypi/cupy-cuda12x/json) | 14.2.0 | `>=3.10` | 14.2.0 | gpu, rapids | `>=13.0`. Keep CUDA extra: CUDA 12 runtime required; excluded from the CPU installation. |
| [cycler](https://pypi.org/pypi/cycler/json) | 0.12.1 | `>=3.8` | 0.12.1 | default | `<1,>=0.10`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [czifile](https://pypi.org/pypi/czifile/json) | 2026.10.14 | `>=3.12` | 2019.7.2.2 | default, all, czi | `unbounded`. Remain unbounded; upstream Python/dependency metadata selects compatible stable. |
| [dinocell](https://pypi.org/pypi/dinocell/json) | 0.74 | `unspecified` | 0.74 | dinocell | `<1.0,>=0.74`. Keep extra: exact Cellpose 4.0.9 / NumPy 2.4.3 / Torch 2.10.0 / Hub 1.7.1 pins contradict newest core. |
| [docutils](https://pypi.org/pypi/docutils/json) | 0.23 | `>=3.9` | 0.23 | dev | `<0.24,>=0.20.1`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [duckdb](https://pypi.org/pypi/duckdb/json) | 2.0.0 | `>=3.11.0` | 1.5.6 | default, databases | `<3,>=1.1`. Promote; raise API-family guard to <3 after DuckDB 2 Parquet/backend comparisons. |
| [faiss-cpu](https://pypi.org/pypi/faiss-cpu/json) | 1.15.1 | `>=3.10` | 1.15.1 | default, search | `<2,>=1.7.4`. Promote; on Mac Python 3.14 require Darwin >=23 (macOS 14), matching available cp314 wheel deployment tags. Other default platforms/interpreters unchanged. |
| [fastremap](https://pypi.org/pypi/fastremap/json) | 1.21.0 | `<4.0,>=3.9` | 1.21.0 | default | `>=1.14.1`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [fsspec](https://pypi.org/pypi/fsspec/json) | 2026.9.0 | `>=3.10` | 2026.9.0 | default, cloud | `>=2023.1`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [gcsfs](https://pypi.org/pypi/gcsfs/json) | 2026.10.1 | `>=3.10` | 2026.10.1 | default, cloud | `>=2023.1`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [glum](https://pypi.org/pypi/glum/json) | 3.4.1 | `>=3.10` | 3.4.1 | default, glum | `<4,>=3.1.2`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [gputil](https://pypi.org/pypi/gputil/json) | 1.4.0 | `unspecified` | 1.4.0 | default | `<2.0,>=1.4.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [gpytorch](https://pypi.org/pypi/gpytorch/json) | 1.15.2 | `>=3.10` | 1.15.2 | default, gpytorch | `<2,>=1.11`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [huggingface-hub](https://pypi.org/pypi/huggingface-hub/json) | 2.2.0 | `>=3.10.0` | 2.2.0 | default | `<3,>=0.25`. Raise cap to <3; saved login now uses supported get_token API; real saved-file regression passes with Hub 2.2. |
| [hypothesis](https://pypi.org/pypi/hypothesis/json) | 6.170.5 | `>=3.11` | 6.168.5 | dev | `<7,>=6.100`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [imageio](https://pypi.org/pypi/imageio/json) | 2.38.1 | `>=3.11` | 2.38.0 | default | `<3.0,>=2.34.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [intel-extension-for-pytorch](https://pypi.org/pypi/intel-extension-for-pytorch/json) | 2.8.0 | `unspecified` | 2.8.0 | intel-gpu | `>=2.1`. Keep vendor extra: 2.8 extension ABI targets Torch 2.8, not tested CPU Torch 2.14.1; upstream recommends plain Torch. |
| [IPython](https://pypi.org/pypi/IPython/json) | 9.17.1 | `>=3.11` | 8.39.0 | default | `<10,>=8.18.1`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [ipywidgets](https://pypi.org/pypi/ipywidgets/json) | 8.1.9 | `>=3.7` | 8.1.9 | default | `<9.0,>=8.1.2`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [jax](https://pypi.org/pypi/jax/json) | 0.12.0 | `>=3.12` | 0.6.2 | default, numpyro | `<1.0,>=0.4`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [joblib](https://pypi.org/pypi/joblib/json) | 1.6.0 | `>=3.10` | 1.6.0 | default | `<2.0,>=1.2`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [keyring](https://pypi.org/pypi/keyring/json) | 25.7.0 | `>=3.9` | 25.7.0 | default, notify | `<27,>=23.0`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [lifelines](https://pypi.org/pypi/lifelines/json) | 0.30.3 | `>=3.11` | 0.30.0 | survival | `<1,>=0.27`. Keep extra: latest pandas <3 requirement contradicts default pandas 3; do not downgrade core. |
| [liffile](https://pypi.org/pypi/liffile/json) | 2026.10.14 | `>=3.12` | 2025.5.10 | default, vendor-watch | `>=2025.1.25`. Promote Python >=3.10 using upstream interpreter filtering: 2025.5.10 on 3.10, current on >=3.12. |
| [lightgbm](https://pypi.org/pypi/lightgbm/json) | 4.7.0 | `>=3.10` | 4.7.0 | default, all, boosting | `<5.0,>=4.0`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [llvmlite](https://pypi.org/pypi/llvmlite/json) | 0.50.0 | `>=3.10` | 0.50.0 | default | `<0.46,>=0.43` / `<1.0,>=0.43`. Retain Intel Mac <0.46: 0.46+ has no macOS x86_64 wheel; normal profile admits 0.50. |
| [lxml](https://pypi.org/pypi/lxml/json) | 6.1.3 | `>=3.8` | 6.1.3 | default | `>=5.1.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [mahotas](https://pypi.org/pypi/mahotas/json) | 1.4.19 | `>=3.10` | 1.4.19 | default, all, zernike | `<2.0,>=1.4.13` / `<2.0,>=1.4.19`. Promote >=1.4.19: new cp313/cp314 wheels remove the old marker; real Zernike calculation verified. |
| [markdown-it-py](https://pypi.org/pypi/markdown-it-py/json) | 4.2.0 | `>=3.10` | 4.2.0 | dev | `<5,>=2.2`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [matplotlib](https://pypi.org/pypi/matplotlib/json) | 3.11.2 | `>=3.11` | 3.10.9 | default | `<4.0,>=3.8.3`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [matplotlib_venn](https://pypi.org/pypi/matplotlib_venn/json) | 1.1.2 | `unspecified` | 1.1.2 | default | `<2.0,>=1.1`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [moderngl](https://pypi.org/pypi/moderngl/json) | 5.13.0 | `>=3.7` | 5.13.0 | default, flow-gpu | `<6,>=5.12`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [mpmath](https://pypi.org/pypi/mpmath/json) | 1.4.1 | `>=3.9` | 1.4.1 | default | `<2,>=1.3`. Keep broad <2; GPyTorch itself selects <=1.3.0. No duplicate spaCR cap. |
| [mrcfile](https://pypi.org/pypi/mrcfile/json) | 1.5.4 | `unspecified` | 1.5.4 | default, all, tem | `<2,>=1.5`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [mypy](https://pypi.org/pypi/mypy/json) | 2.4.0 | `>=3.10` | 2.4.0 | dev | `<3,>=1.11`. Keep dev; raise API-family guard to <3 to admit current 2.x. |
| [napari](https://pypi.org/pypi/napari/json) | 0.9.2 | `>=3.11` | 0.7.1 | default, napari | `<1.0,>=0.5`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [natsort](https://pypi.org/pypi/natsort/json) | 8.4.0 | `>=3.7` | 8.4.0 | default | `<9.0,>=8.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [nd2](https://pypi.org/pypi/nd2/json) | 0.12.0 | `>=3.10` | 0.12.0 | default, vendor-watch | `<1,>=0.10`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [nd2reader](https://pypi.org/pypi/nd2reader/json) | 3.3.1 | `>=3.6` | 3.3.1 | default, all, nd2 | `<4.0,>=3.3.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [numba](https://pypi.org/pypi/numba/json) | 0.68.0 | `>=3.10` | 0.68.0 | default | `<0.63,>=0.60` / `<1.0,>=0.60`. Retain Intel Mac <0.63 wheel boundary; normal profile admits 0.68. |
| [numcodecs](https://pypi.org/pypi/numcodecs/json) | 0.17.1 | `>=3.12` | 0.13.1 | default, zarr | `<1,>=0.12`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [numexpr](https://pypi.org/pypi/numexpr/json) | 2.14.2 | `>=3.11` | 2.14.1 | default | `<3.0,>=2.8.4`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [numpy](https://pypi.org/pypi/numpy/json) | 2.5.4 | `>=3.12` | 2.2.6 | default | `<2.0,>=1.26.4` / `<3.0,>=1.26.4`. Retain <3 API-family guard; Intel Mac <2 below Python 3.13 is coupled to available Torch/OpenCV wheels. Latest 2.5.4 exercised. |
| [numpyro](https://pypi.org/pypi/numpyro/json) | 0.22.0 | `>=3.11` | 0.20.0 | default, numpyro | `<1.0,>=0.13`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [nvidia-ml-py](https://pypi.org/pypi/nvidia-ml-py/json) | 13.615.71 | `unspecified` | 13.615.71 | default | `<14,>=11.450.51`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [omero-py](https://pypi.org/pypi/omero-py/json) | 5.23.0 | `>=3.10` | 5.23.0 | omero | `<6,>=5.17`. Keep extra: ZeroC Ice <3.7 builds fail on CPython 3.12 (removed eval.h); not an installable modern default. |
| [opencv-python](https://pypi.org/pypi/opencv-python/json) | 5.0.0.93 | `>=3.6` | 5.0.0.93 | full | `unbounded`. Keep alternate full extra: owns the same cv2 files as core headless OpenCV; upstream forbids co-installation. |
| [opencv-python-headless](https://pypi.org/pypi/opencv-python-headless/json) | 5.0.0.93 | `>=3.6` | 5.0.0.93 | default, headless | `<4.12,>=4.9.0.80` / `<6.0,>=4.9.0.80`. Raise normal cap to <6 after OpenCV 5 imports/behavior; retain Intel Mac <4.12 to avoid NumPy 2-only wheel dependency. |
| [openpyxl](https://pypi.org/pypi/openpyxl/json) | 3.1.5 | `>=3.8` | 3.1.5 | default | `<4.0,>=3.1`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [packaging](https://pypi.org/pypi/packaging/json) | 26.3 | `>=3.9` | 26.3 | default | `>=21.3`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [pandas](https://pypi.org/pypi/pandas/json) | 3.0.6 | `>=3.11` | 2.3.3 | default | `<4.0,>=2.2.1`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [paramiko](https://pypi.org/pypi/paramiko/json) | 5.1.0 | `>=3.9` | 5.1.0 | default | `<6,>=3.5`. Raise API-family guard to <6; 5.1 imports in clean modern default. |
| [patsy](https://pypi.org/pypi/patsy/json) | 1.0.3 | `>=3.6` | 1.0.3 | default | `<2.0,>=0.5.6`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [pdfplumber](https://pypi.org/pypi/pdfplumber/json) | 0.11.10 | `>=3.8` | 0.11.10 | default, papers | `<1,>=0.11`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [pillow](https://pypi.org/pypi/pillow/json) | 13.0.0 | `>=3.11` | 12.3.0 | default | `<14,>=10.2.0`. Raise API-family guard to <14 after 13.0 imports and image/attribution behavior. |
| [pingouin](https://pypi.org/pypi/pingouin/json) | 0.7.0 | `>=3.11` | 0.6.1 | dev | `<2.0,>=0.5.5`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [piper-tts](https://pypi.org/pypi/piper-tts/json) | 1.8.0 | `>=3.9` | 1.8.0 | default, all, tutorial | `<2,>=1.2`. Promote; retain Intel Mac Python 3.14 exclusion. Mac arm64 Python 3.14 requires Darwin >=23 (macOS 14) for ONNX Runtime wheels. Import checked without synthesis/model download. |
| [protobuf](https://pypi.org/pypi/protobuf/json) | 7.36.2 | `>=3.10` | 7.36.2 | default | `>=5.28.3`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [psutil](https://pypi.org/pypi/psutil/json) | 7.2.2 | `>=3.6` | 7.2.2 | default | `<8,>=5.9.8`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [psycopg](https://pypi.org/pypi/psycopg/json) | 3.3.6 | `>=3.10` | 3.3.6 | default, databases | `<4,>=3.1`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [pyarrow](https://pypi.org/pypi/pyarrow/json) | 26.0.0 | `>=3.11` | 25.0.1 | default, dev | `<27,>=14.0.2`. Promote runtime Parquet backend from dev; raise guard to <27 after PyArrow 26 Parquet round-trips. |
| [pycromanager](https://pypi.org/pypi/pycromanager/json) | 1.0.2 | `>=3.6` | 1.0.2 | default, microscope | `<2,>=0.28`. Promote; raise API-family guard to <2; installed 1.0 adapter imports without requiring microscope connection. |
| [pyfixest](https://pypi.org/pypi/pyfixest/json) | 0.60.0 | `>=3.10` | 0.60.0 | default, pyfixest | `<1,>=0.40.1`. Retain bounds; exclude default only on Mac Python 3.10 because no compatible Mac cp310 wheel exists at or above the API floor. Keep the optional compiler-install alias. |
| [pylibCZIrw](https://pypi.org/pypi/pylibCZIrw/json) | 6.1.0 | `<3.15,>=3.9` | 6.1.0 | default, all, czi | `<7.0,>=5.0.0`. Promote below Python 3.14 except Intel Mac: no cp314 or macOS x86_64 wheels in 6.1 release. |
| [pymc](https://pypi.org/pypi/pymc/json) | 6.3.2 | `>=3.12` | 5.26.0 | default, pymc | `<7,>=5.10`. Promote; raise API-family guard to <7 after PyMC 6 compiled normal log-density check; compatible 5.x resolves on Python 3.10. |
| [PyOpenGL](https://pypi.org/pypi/PyOpenGL/json) | 3.1.10 | `unspecified` | 3.1.10 | default | `<4,>=3.1.7`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [pypdf](https://pypi.org/pypi/pypdf/json) | 6.21.2 | `>=3.9` | 6.21.2 | default | `<7.0,>=6.16.1`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [pypdfium2](https://pypi.org/pypi/pypdfium2/json) | 5.14.0 | `>=3.6` | 5.14.0 | dev | `<6,>=5.13.0`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [pyqtgraph](https://pypi.org/pypi/pyqtgraph/json) | 0.14.0 | `>=3.10` | 0.14.0 | default, all, qt | `<1,>=0.13.3`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [PySide6](https://pypi.org/pypi/PySide6/json) | 6.12.0 | `<3.16,>=3.10` | 6.12.0 | default, all, flowview, qt, tutorial | `!=6.12.0,<7,>=6.6`. Retain only !=6.12.0 and <7: Python 3.10 painter fatal reference-loss reproduction; 10,000-frame 6.11.2 control passes. 6.12.1 remains allowed. |
| [pytest](https://pypi.org/pypi/pytest/json) | 9.1.1 | `>=3.10` | 9.1.1 | dev | `<10,>=8.0`. Keep dev; raise API-family guard to <10 after pytest 9 runs. |
| [pytest-qt](https://pypi.org/pypi/pytest-qt/json) | 4.5.0 | `>=3.9` | 4.5.0 | dev | `<5,>=4.4`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [pytest-xdist](https://pypi.org/pypi/pytest-xdist/json) | 3.8.0 | `>=3.9` | 3.8.0 | dev | `<4,>=3.6.1`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [PyWavelets](https://pypi.org/pypi/PyWavelets/json) | 1.10.0 | `>=3.12` | 1.8.0 | default | `<2,>=1.4`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [qtawesome](https://pypi.org/pypi/qtawesome/json) | 1.4.2 | `>=3.7` | 1.4.2 | default, all, qt, tutorial | `<2,>=1.3`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [rapidocr-onnxruntime](https://pypi.org/pypi/rapidocr-onnxruntime/json) | 1.4.4 | `<3.13,>=3.6` | 1.4.4 | papers | `<2,>=1.3`. Keep extra: requires full opencv-python; additionally latest supports Python <3.13. |
| [rdkit](https://pypi.org/pypi/rdkit/json) | 2026.9.1 | `unspecified` | 2026.9.1 | default, chem | `>=2023.3`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [readlif](https://pypi.org/pypi/readlif/json) | 0.6.6 | `<4.0,>=3.9` | 0.6.6 | default, all, lif | `unbounded`. Remain unbounded; upstream Python/dependency metadata selects compatible stable. |
| [readme-renderer](https://pypi.org/pypi/readme-renderer/json) | 46.0 | `>=3.10` | 46.0 | dev | `<47,>=43`. Keep dev; raise guard to <47 to admit current 46. |
| [requests](https://pypi.org/pypi/requests/json) | 2.34.2 | `>=3.10` | 2.34.2 | default | `<3.0,>=2.28`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [ruff](https://pypi.org/pypi/ruff/json) | 0.17.1 | `>=3.7` | 0.17.1 | dev | `<1,>=0.9`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [s3fs](https://pypi.org/pypi/s3fs/json) | 2026.9.0 | `>=3.10` | 2026.9.0 | default, cloud | `>=2023.1`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [samcell](https://pypi.org/pypi/samcell/json) | 1.2.0 | `>=3.8` | 1.2.0 | default, samcell | `<2.0,>=1.2`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [scikit-image](https://pypi.org/pypi/scikit-image/json) | 0.26.0 | `>=3.11` | 0.25.2 | default | `<0.28,>=0.22.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [scikit-learn](https://pypi.org/pypi/scikit-learn/json) | 1.9.1 | `>=3.11` | 1.7.2 | default | `<2.0,>=1.5.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [scikit-posthocs](https://pypi.org/pypi/scikit-posthocs/json) | 0.17.1 | `>=3.9` | 0.17.1 | default | `<0.20,>=0.10.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [scipy](https://pypi.org/pypi/scipy/json) | 1.18.1 | `>=3.12` | 1.15.3 | default | `<2.0,>=1.12.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [seaborn](https://pypi.org/pypi/seaborn/json) | 0.13.2 | `>=3.8` | 0.13.2 | default | `<1.0,>=0.13.2`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [shap](https://pypi.org/pypi/shap/json) | 0.54.1 | `>=3.12` | 0.49.1 | default | `<1.0,>=0.47.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [statsmodels](https://pypi.org/pypi/statsmodels/json) | 0.15.0 | `>=3.10` | 0.15.0 | default | `<0.16,>=0.14.0`. Raise minor-family cap to <0.16 after actual 0.15 OLS/WLS and alternate-backend numeric comparisons. |
| [tables](https://pypi.org/pypi/tables/json) | 3.11.1 | `>=3.11` | 3.10.1 | default | `<4.0,>=3.8.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [tensorboard](https://pypi.org/pypi/tensorboard/json) | 2.21.0 | `>=3.9` | 2.21.0 | default | `<3.0,>=2.21`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [threadpoolctl](https://pypi.org/pypi/threadpoolctl/json) | 3.7.0 | `>=3.9` | 3.7.0 | default, sweep | `<4,>=3.0`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [tifffile](https://pypi.org/pypi/tifffile/json) | 2026.10.15 | `>=3.12` | 2025.5.10 | default | `>=2023.4.12`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [timm](https://pypi.org/pypi/timm/json) | 1.0.30 | `>=3.8` | 1.0.30 | default, embeddings | `<2.0,>=0.9`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [tomli](https://pypi.org/pypi/tomli/json) | 2.5.0 | `>=3.8` | 2.5.0 | dev | `>=2.0`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [torch](https://pypi.org/pypi/torch/json) | 2.14.1 | `>=3.10` | 2.14.1 | default | `<3.0,>=2.0`. Keep <3 and Intel Mac wheel marker; CPU index profile preserved; real CPU inference verified. |
| [torch-directml](https://pypi.org/pypi/torch-directml/json) | none (0.2.5.dev240914 dev only) | `>=3.7` | none | directml | `>=0.2`. Keep Windows-only vendor extra: no stable release; published development wheel pins the older Torch/torchvision pair. |
| [torchcam](https://pypi.org/pypi/torchcam/json) | 0.5.0 | `<4,>=3.11` | 0.4.0 | default, attribution | `<1.0,>=0.4.0` / `<1.0,>=0.5`. Promote >=0.5 on Python >=3.11: 0.5 supports NumPy 2; Python 3.10-compatible 0.4 requires NumPy <2 and stays optional. |
| [torchvision](https://pypi.org/pypi/torchvision/json) | 0.29.1 | `!=3.14.1,>=3.10` | 0.29.1 | default | `<1.0,>=0.15`. Keep <1 and Intel Mac wheel marker; upstream exact Torch pairing controls selection; real CPU NMS verified. |
| [tqdm](https://pypi.org/pypi/tqdm/json) | 4.70.1 | `>=3.8` | 4.70.1 | default | `>=4.65.0`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [trackastra](https://pypi.org/pypi/trackastra/json) | 0.5.6 | `>=3.10` | 0.5.6 | default, all, trackastra | `<1.0,>=0.5`. Promote with existing Intel Mac Torch marker retained; configuration import verified without downloading weights. |
| [trackpy](https://pypi.org/pypi/trackpy/json) | 0.7 | `>=3.8` | 0.7 | default | `<1.0,>=0.6.2`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [transformers](https://pypi.org/pypi/transformers/json) | 5.20.2 | `>=3.10.0` | 5.20.2 | default, foundation | `<6,>=4.40`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |
| [ultrack](https://pypi.org/pypi/ultrack/json) | 0.8.0 | `<3.14,>=3.11` | 0.7.2 | default, all, ultrack | `<1.0,>=0.6`. Promote Python >=3.10,<3.14 on Higra-supported platforms. Linux requires x86_64/AMD64: Higra has no Linux ARM distribution or sdist. Mac ARM and Windows remain enabled. Joint Python 3.10 solve uses 0.6.3 with Zarr 2; Python 3.12 uses 0.8. |
| [ultralytics](https://pypi.org/pypi/ultralytics/json) | 8.4.183 | `>=3.8` | 8.4.183 | all, papers, plaque | `<9,>=8.0` / `<9,>=8.4`. Keep extra: requires full opencv-python, which collides with core headless cv2 distribution. |
| [umap-learn](https://pypi.org/pypi/umap-learn/json) | 0.5.12 | `>=3.9` | 0.5.12 | default, umap | `<1.0,>=0.5.11`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [vispy](https://pypi.org/pypi/vispy/json) | 0.17.0 | `>=3.10` | 0.17.0 | default, all, fractal | `<1.0,>=0.14`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [wgpu](https://pypi.org/pypi/wgpu/json) | 0.32.0 | `>=3.11` | 0.31.0 | default | `<0.33,>=0.25` / `<=0.31.0,>=0.25`. Keep existing >=3.10 core marker: latest >=3.11, upstream selects 0.31 on Python 3.10; no GPU initialization. |
| [win10toast](https://pypi.org/pypi/win10toast/json) | 0.9 | `unspecified` | 0.9 | default, all, qt, tutorial | `>=0.9`. Keep Windows-only core marker; no Linux import claim. |
| [xenon](https://pypi.org/pypi/xenon/json) | 0.9.3 | `unspecified` | 0.9.3 | dev | `<1,>=0.9`. Keep development/test/build tool; retain floor and guard admitting latest compatible stable. |
| [xgboost](https://pypi.org/pypi/xgboost/json) | 3.4.1 | `>=3.12` | 3.2.0 | default | `<4,>=2.0.3`. Retain existing floor and API-family guard admitting latest compatible stable. |
| [zarr](https://pypi.org/pypi/zarr/json) | 3.4.1 | `>=3.12` | 2.18.3 | default, zarr | `<4,>=2.16`. Promote runtime feature; retain floor and current API-family guard; admits latest compatible stable. |

### Linux ARM64 dependency correction (2026-10-21)

Hosted Ubuntu 24.04 ARM64/Python 3.11 job 119488634092 failed the wheel
`[qt,dev,zernike]` installation because Ultrack requires Higra >=0.6.10.
[Higra's official release metadata](https://pypi.org/pypi/higra/json) publishes
Linux x86_64, macOS x86_64/ARM64 and Windows x86_64 wheels, with no Linux ARM
wheel or source distribution. Ultrack therefore follows Higra's Linux x86_64
architecture limit in the default, named and aggregate profiles. This preserves
Mac ARM64, Windows and x86 Linux installations and all existing core bounds.

The corrected default plus `qt,dev,zernike` requirements resolve 376 packages for
`aarch64-manylinux_2_39`, CPython 3.11.17, matching the hosted Ubuntu 24.04 image,
with CPU Torch. This uses the actual ordinary-install source policy: Mahotas
may build from its source release, as in the hosted job. A separate wheel-only
claim would be false for that ARM profile. Nine owning marker/profile checks
pass. Native ARM execution remains for hosted CI; this is target resolution.

### Target-platform default resolution

The final active defaults were resolved with uv 0.13.4 using its documented
[`--python-platform` / `--python-version` target support](https://docs.astral.sh/uv/concepts/resolution/),
CPU Torch, `--only-binary :all:` and the existing pure-Python exceptions.
Two additional pure-Python exceptions are necessary for the Python 3.10
Ultrack/Zarr graph: [websocket 0.2.1](https://pypi.org/pypi/websocket/0.2.1/json)
and [asciitree 0.3.3](https://pypi.org/pypi/asciitree/0.3.3/json).
Their official sdists were inspected: no C/C++/Cython sources or extension-module
build declarations, only Python/data copied by setuptools. This does not exempt
compiled dependencies such as gevent or greenlet from the wheel requirement.

| Target / minimum system | Python 3.10.22 | Python 3.12.14 | Python 3.14.7 |
| --- | --- | --- | --- |
| Windows x86_64 | resolves | resolves | resolves |
| macOS 13 arm64 | resolves | resolves | resolves; Faiss, Piper and btrack optional |
| macOS 14 arm64 | same profile (inferred from macOS 13) | same profile (inferred from macOS 13) | resolves; btrack optional |
| macOS 15 arm64 | same profile (inferred from macOS 13) | same profile (inferred from macOS 13) | resolves; all three native features default |
| macOS 13 Intel | resolves | resolves | existing core fails: Numba <0.63 has no cp314 wheel |

The Mac Python 3.14 deployment limits follow official
[CVXOPT cp314 wheel tags](https://pypi.org/pypi/cvxopt/1.3.3/json),
[Faiss wheel tags](https://pypi.org/pypi/faiss-cpu/1.15.1/json), and
[ONNX Runtime wheel tags](https://pypi.org/pypi/onnxruntime/1.31.0/json).

The Intel Python 3.14 failure was reproduced with the pre-promotion core, so it
is an existing unsupported wheel combination rather than a new restriction.
No Intel Python 3.14 success is claimed. All compatible Python 3.10 core
NumPy/pandas/Cellpose profiles remain resolvable. Intel Mac selects Torch 2.2.2,
NumPy 1.26.4 and compatible JAX/PyMC through the preserved wheel/core constraints.
The Linux Python 3.10 compiled-wheel policy also resolves with these exceptions.

For the Darwin-release markers, the project requirements were evaluated against
the complete target environment before passing the active requirements to uv.
`MACOSX_DEPLOYMENT_TARGET` selects matching wheel deployment tags. This avoids
assuming uv's generic Darwin target supplies a real machine's kernel release.
The exact defaults remain enabled on newer Mac systems; aliases permit explicit
native builds on older systems. These are resolver checks, not native execution
or GUI/scientific acceptance on Windows or macOS.

## Historical decisions

The following notes retain the context of earlier implementations. Current
declarations and the dated decisions above take precedence.

## Module level

### lines 1-29

```python
import sys
```

setup.py must do nothing but describe the package.

It used to end with a module-level loop that shelled out to

`subprocess.run(['pip', 'install', dep])`. That ran on every build and on every `pip install .`, which broke PEP 517 isolated builds (no pip inside the isolated env), broke every offline/air-gapped install, swallowed all failures behind a bare `except CalledProcessError: pass`, invoked the bare name `pip` (absent from PATH in many venv layouts), and installed an entire second, unused Qt binding: pyqtgraph, pyqt6, pyqt6.sip, qtpy and superqt — a hand-copy of cellpose's `gui` extra, duplicated within itself. Files under spacr/ import PySide6; zero import PyQt6, qtpy or superqt. The block is gone; nothing replaced it, because nothing needed it.

PYQTGRAPH IS NO LONGER PART OF THAT BAN, and the paragraph above is left standing because it was true when it was written. spacr/qt/widgets/ fast_plots.py is the volcano, the Q-Q, the p-histogram, the control panel and the guide-agreement plot, and the regression results panel builds all five -- so pyqtgraph is now a first-class dependency spaCR imports, declared in the core requirements beside PySide6. It is not a Qt BINDING; it draws through whichever binding is installed, so none of the ABI reasoning that banned PyQt6 applies to it. Reported from a real install that had PySide6 and not this, where opening ANY module raised RuntimeError out of the screen factory.

Project metadata that PEP 621 owns (name, requires-python, classifiers, authors, URLs) now lives in pyproject.toml. The fields below stay here because packaging/build_{debian,macos,windows}.* and four tests read this file as text or execute it directly; pyproject.toml declares them `dynamic` so setuptools takes them from this call.

### lines 37-175

```python
dependencies = [
```

Every bound below is meant to be evidence, not convention: a floor names an API spaCR actually calls, a ceiling names a break that was observed. Bounds audited end-to-end on 2026-07-26 against the PyPI JSON API and against two throwaway conda envs (CPython 3.12 and 3.13) built with the widened set. The audit changed bounds in three directions, and the direction matters:

ceilings LOWERED, because the old one was too loose and admitted a resolve that breaks spaCR at runtime (pingouin, scikit-image, statsmodels). These were the most valuable findings in the audit: `<1.0` on a 0.x package is decorative, because 0.x breaks at the MINOR. ceilings RAISED, because no break exists above them (pillow, psutil, IPython, lxml, xgboost, protobuf). floors RAISED, because the old one admitted a resolve that is already broken (scikit-learn) or was simply false (torchvision).

Several floors are *higher* than the oldest release that would work — e.g. nothing spaCR calls needs matplotlib 3.8.3 (the newest API used, indexing `matplotlib.colormaps`, is 3.5), scikit-image 0.22 (the newest is `regionprops_table(spacing=)`, 0.20), pandas 2.2.1 (the newest is `DataFrame.attrs`, 1.0) or scipy 1.12. They are left alone deliberately: an over-tight floor only costs something when it blocks a resolution, and none of these do on any Python in `requires-python`. Lowering them would buy nothing and would admit combinations no spaCR test has ever run.

2026-07-27: the list was reconciled against what spaCR actually imports.

The method, re-run independently rather than inherited: `ast.parse` every one of the 159 files under `spacr/`, `ast.walk` each tree (so imports inside functions, methods, `try`/`except` bodies and `if TYPE_CHECKING` all count, not just module scope), collect every top-level module name, then resolve each declared distribution to the import names it actually installs by reading the *installed* `top_level.txt` / RECORD rather than guessing from the project name — that is what catches opencv-python-headless -> `cv2`, pillow -> `PIL`, biopython -> `Bio`, gputil -> `GPUtil`. Every candidate was then re-checked with a raw `grep -rIn -w` over the whole tree, non-Python files included, to catch dynamic references a syntax tree cannot see.

That last step mattered exactly once, and it is why the census is run rather than trusted: `umap-learn` has ZERO import statements anywhere in spaCR. It is reached through `umap = _LazyModule('umap.umap_', ...)` at spacr/utils.py:197 — the module name is a *string literal*, invisible to an import census. It is a core dependency and stays one. Any future pass over this list has to grep for the string form before believing a zero.

18 distributions had zero imports, zero string references and zero dynamic references, and are gone: transformers, monai, segmentation_models_pytorch, torch-geometric, PyWavelets, rapidfuzz, wandb, gdown, pytz, ipykernel, ttf_opensans (spaCR bundles its own Open Sans under spacr/resources/font/), brokenaxes, gpustat, openai, keyring and importlib-metadata. Three of those had a *prose* hit and no code hit — "transformers" in a GUI blurb, "@openai/codex" in an npm one-liner, "OS keyring" in a stale docstring — which is precisely the difference a raw grep has to be read for rather than counted.

9 distributions were imported and never declared. Eight are still declared; the ninth outlived its own import and is gone again, which is recorded below rather than quietly dropped. Five were module-scope, unguarded imports, so the install was only ever working by accident — they arrived transitively and would vanish the day the package that dragged them in changed its own mind: requests   spacr/utils.py:2 and spacr/gui_utils.py:1 (module scope). Arrived via huggingface-hub, which is exactly the package that drops requests for httpx at 1.0 — see the hub note. joblib     spacr/utils.py:207 (module scope), spacr/object.py:49. Arrived via scikit-learn. natsort    spacr/submodules.py:36 (module scope). Arrived via cellpose. patsy      spacr/ml.py:33 (module scope). Arrived via statsmodels. sympy      WAS spacr/gui_elements.py:26 (module scope), via torch. That file left with the Tkinter interface and took the only import of sympy with it, so sympy is NO LONGER DECLARED — see the `sympy` REMOVED note further down. This line used to cite a file that has not existed for some time. and four are function-local and already guarded, so they are declared for honesty rather than to fix a break: nvidia-ml-py (imported as ``pynvml`` BOTH distributions install a module of that name -- inside spacr/qt/widgets/home.py's `_nvml()`, one site as of 2026-09-14, not the two this line used to cite), win10toast (spacr/qt/notify.py:56, behind `if system == "Windows"`, so it carries the marker), and catboost/lightgbm, which are alternative model backends behind a `model_type=` string and live in the `boosting` extra — see the note there.

WHAT "THE OPTIMAL DEPENDENCY VERSIONS" MEANS.

The phrase has two readings that imply opposite CI jobs -- the oldest set that passes, or the newest set that passes -- and a list edited under both readings at once drifts in both directions. spaCR means the NEWEST. Both jobs exist because they answer different questions, and only one of them is a recommendation.

WHAT A USER GETS is the newest. `pip install spacr` resolves the TOP of every range below, so the upper end of each range is the version spaCR recommends. The `Fast / Full suite control` job in .github/workflows/tests.yml installs with no constraints file at all and runs the suite against whatever pip picks that day, which is the only way the recommendation stays true as new releases land.

WHAT A USER IS PROMISED is the floor. A lower bound is for an environment pinned by something else -- a cluster module, a conda solve, another package's own cap. It is not advice, and it is worthless unless something installs it, so the `Minimum dependencies` job in the same workflow installs .github/constraints/minimum-py39.txt on CPython 3.9 and runs the same suite there.

A lower bound is therefore THE OLDEST VERSION CI ACTUALLY INSTALLS AND TESTS, never a guess at the oldest that might work. Raising one is cheap; leaving one unverified is how "it installed and then crashed" reports are made, because pip resolves a low floor happily and the failure lands on a user. cellpose is the worked example: `>=4.0` resolved to 4.0.1 in CI, whose `CellposeModel` signature spaCR has never been developed against, so the floor moved to 4.0.7 rather than the contract test being loosened.

Every constraint pin equals its declared floor exactly, unless the minimum profile cannot install that combination -- torchvision pins torch to an EXACT release, so the two move as a matched pair rather than each sitting on its own floor, and umap-learn and tensorboard drag scikit-learn and protobuf up -- and then the constraints file names the package and says why. A pin quietly sitting above its floor leaves the floor the one bound in this file that nothing tests.

An upper bound is a MAJOR VERSION spaCR HAS NOT SEEN. It exists so a breaking release cannot arrive silently between a user's `pip install` and their first run, and it is meant to be raised deliberately after testing. A cap left to rot denies users exactly the newest version this policy is about. Where a package has no upper bound that is a considered choice, not an oversight: tifffile, lxml, fastremap, tqdm and protobuf have not broken spaCR across a major.

"Newest" deliberately does NOT mean a lockfile. Measured rather than assumed: on Python 3.9 the newest resolvable IPython is 8.18.1 while on 3.11+ it is 9.16.1, and matplotlib is 3.9.4 versus 3.11.1. Across the six interpreters spaCR supports there is no single newest-that-works set, so any file claiming one would be wrong on 3.9 the day it was written.

tests/test_optimal_dependency_versions.py holds this shape: both ends blocking, the floor end constrained, the newest end unconstrained, and no unexplained pin above a floor.

### lines 177-211

```python
'numpy>=1.26.4,<3.0',
```

`<2.0` used to be the single most expensive bound in this file: numpy 1.26.4 was the ONLY release satisfying it, its wheels stop at cp312, and that is what forced `requires-python = ">=3.10,<3.13"`.

It is now `<3.0`, and every one of the three blockers that held it was closed rather than argued away:

1. `np.trapz` (removed in numpy 2.0) is gone from all three call sites. spacr/utils.py:7 and spacr/attribution.py:66 now resolve `_trapezoid = getattr(np, 'trapezoid', None) or np.trapz` once at module scope, and spacr/timelapse.py:25-28 prefers `numpy.trapezoid` with a `numpy.trapz` fallback for numpy 1.x. The old fallback there went through `scipy.integrate.trapz`, which was already dead — SciPy removed it in 1.14 and `scipy<2.0` resolves 1.18 — so `import spacr.timelapse` was the one module that failed under numpy 2. It no longer is. 2. torchcam 0.4.0 AND 0.4.1 — i.e. every release that satisfied its old pin — declare `numpy<2.0.0`. The pin is spurious: torchcam touches numpy only in an overlay helper (`np.asarray`/`np.uint8`) and was verified running GradCAM correctly under numpy 2.4.4. But pip cannot be argued with, so torchcam has left the core list for the `attribution` extra. It was already imported lazily (spacr/attribution.py:580, inside `_torchcam_cam`), so the core install never needed it. 3. tests/test_diameter_estimator.py:1083 called the `ndarray.ptp()` method*, removed in 2.0. It now calls `np.ptp(field)`, matching the five other sites that already used the function form.

The floor stays at 1.26.4 rather than moving up with the ceiling: it is the oldest release with cp312 wheels, it costs nothing on 3.13 (where pip picks 2.3+ regardless), and dropping it would gratuitously break anyone pinned to the 1.x line who is happy there. On 3.10 this resolves numpy 2.2.6 (the last line with cp310 wheels); on 3.13, 2.3+.

### lines 213-221

```python
'pandas>=2.2.1,<4.0',
```

`<3.0` is a real ceiling, not a convention. pandas 3.0 makes

Copy-on-Write the only mode, which turns the five chained-inplace calls at spacr/submodules.py:1415,1416,1429,1430,1468 into silent no-ops (reproduced: the inf/NaN cleanup simply does not happen, and ChainedAssignmentError is a Warning, so execution continues with wrong data). It also drops `include_groups` from `groupby().apply`, which makes spacr/plot.py:3064 hand seaborn a frame missing its `x_column`, and forbids `read_html` on a raw string, which is exactly what spacr/sim.py:655 passes.

### lines 224-254

```python
'cellpose>=4.0.7,<5.0',
```

4.0.7, not 4.0: the floor has to be a version the test suite actually passes on. `CellposeModel.__init__` gained `use_bfloat16` and `eval`'s `resample` default changed between 4.0.1 and 4.0.7, and tests/test_cellpose_api_contract.py records the installed signature and asserts it matches -- so the minimum-dependencies CI job, which resolves the floor literally, failed ~22 tests against an API this project has never been developed on.

Declaring a floor nothing verifies is how "it installed and then crashed" reports are made: pip resolves to it happily, the install succeeds, and the failure lands somewhere else entirely.

THE CEILING, settled 2026-08-12. `<5.0` stays, and the range is now actually exercised at BOTH ends: 53 cellpose tests pass on 4.0.7 and on 4.2.1.1. Before that only the floor was, and 4.2 -- which this range has always admitted -- broke two things that a user upgrading would have hit:

MODEL_NAMES grew from ['cpsam'] to ['cpsam_v2','cpdino', 'cpdino-vitb','cpsam'], and `_resolve_cellpose_pretrained` stopped at cpsam -- so the dropdown offered the new models and the run silently segmented with the old weights. `rescale` stopped being ignored, so spaCR's `rescale=False` became `int(200/False)` inside cellpose once `resample` was on.

NOT SPLIT BY PYTHON VERSION, though that was considered: cellpose declares no `requires_python` at any release, and 4.0.7 and 4.2.1.1 have byte-identical `requires_dist`. There is no version of cellpose that needs a different interpreter, so a marker splitting them would invent a constraint upstream does not have. spaCR adapts to whichever is installed instead -- see tests/cellpose_api_contract.py, which records one exact signature per release rather than one tolerant union.

### lines 256-264

```python
'scikit-image>=0.22.0,<0.28',
```

`segment-anything` removed: PyPI's `segment-anything` has exactly one release (1.0, 2023-04-06) with empty author, homepage and summary — Meta never published SAM to PyPI. spaCR imports it in zero files, and cellpose already depends on segment_anything itself. It was an unpinned, unattributed name in the supply chain for no benefit. spaCR uses the replacement APIs introduced before 0.27: `footprint_rectangle`, `max_size`, `opening`, `closing` and `dilation`. Compatibility fallbacks retain 0.22-0.24 support without importing the aliases removed in 0.27. Keep the next minor as the explicit audit gate.

### lines 266-270

```python
'scikit-learn>=1.5.0,<2.0',
```

Floor RAISED from 1.4.1, which admitted a resolve that crashes: spacr/utils.py:6247 calls `TSNE(..., max_iter=1000)`, and `max_iter` did not exist on TSNE before scikit-learn 1.5.0 (it was `n_iter`). On 1.4.x that is `TypeError: TSNE.__init__() got an unexpected keyword argument`. utils.py:6245-6246 already documents the rename; the pin never followed.

### lines 272-275

```python
'scikit-posthocs>=0.10.0,<0.20',
```

Only `sp.posthoc_dunn(val_col=, group_col=, p_adjust=)` is used (spacr/sp_stats.py:184-185, spacr/plot.py:3559). Its signature is byte-identical at 0.10.0, 0.11.4 and 0.14.0, so no ceiling is warranted; `<0.20` names a version that does not exist and never bites.

### lines 277-286

```python
'trackpy>=0.6.2,<1.0',
```

`mahotas` REMOVED from the core list -> the `zernike` extra. Not a numpy-2 problem: 1.4.18 computes Zernike moments correctly under numpy 2.4.4, because its wheels are numpy-2 ABI clean. It is a *wheel* problem. Verified against the PyPI JSON API on 2026-07-27: 1.4.18 publishes cp310, cp311 and cp312 and nothing else — no cp313 has ever been published at any version, and the last aarch64 build was 1.4.13. In the core list it therefore does not merely block Python 3.13, it makes `pip install spacr` on 3.13 attempt a C++ source build, which succeeds only where a toolchain happens to exist. Zernike moments are the sole consumer, so the feature moves with the package.

### lines 288-316

```python
'statsmodels>=0.14.0,<0.15',
```

The deprecated lowercase ``links.logit`` alias is no longer used: spacr/ml.py imports and constructs ``Logit``. Keep the existing 0.15 boundary until the complete statistical suite has been qualified against that release; the warning cleanup itself does not claim broader dependency compatibility.

Floor RAISED from 0.13.0, which was a guess and was wrong. The comment here justified 0.13 by ``statsmodels.othermod.betareg.BetaModel`` — the first API spaCR needed, not the LAST. Three later APIs are used and none of them exists before 0.14.0, which is why the min-deps job (pinned at 0.13.5) failed 16 tests no other job failed:

``links.Identity``  — spacr/ml.py:1142,1154. Before 0.14.0 the CamelCase link classes did not exist for the Power subclasses; 0.13 spells it ``links.identity``, so this is AttributeError, not a deprecation. (``Logit`` and ``Log``, also used, DO predate it — which is exactly why the gap went unnoticed.) ``BetaResults.get_influence`` — spacr/regression_qc.py:1480 asks the fitted model for its own leverage. Added to ``othermod/betareg.py`` in 0.14.0; absent in 0.13.x, where the `getattr` guard there silently falls back to the design matrix and a beta fit is standardised by the wrong hat diagonal. perfect separation in ``GLM._fit_irls`` RAISES ``PerfectSeparationError`` in 0.13 and only WARNS (``PerfectSeparationWarning``) from 0.14.0 on. spaCR's binomial backends rely on the fit returning.

0.14.0 publishes cp39 manylinux wheels, so the floor stays reachable on the oldest interpreter spaCR claims.

### lines 318-322

```python
'patsy>=0.5.6,<2.0',
```

ADDED. `from patsy import dmatrices` at spacr/ml.py:33 is module scope and unguarded, so `import spacr.ml` needs it. It was arriving only because statsmodels declares it; statsmodels 0.15 is expected to finish the move to its own formula engine, and the line above already admits 0.14.x, so this was one upstream release away from breaking.

### lines 324-332

```python
'shap>=0.47.0,<1.0',
```

Floor RAISED from 0.45.0, which was a guess. spacr/sim.py:1541 calls `shap.summary_plot(..., rng=np.random.default_rng(42))` — the seed that makes the beeswarm's jitter reproducible. `summary_plot` IS `shap.plots._beeswarm.summary_legacy`, and that function grew `rng` in 0.47.0: checked the signature at the 0.45.0, 0.46.0 and 0.47.0 tags, and neither of the first two takes it or a `**kwargs` that would swallow it. On 0.45 the call is `TypeError: summary_legacy() got an unexpected keyword argument 'rng'`, which is what the min-deps job hit. 0.47.0 publishes cp39 wheels, so the floor stays reachable on 3.9.

### lines 334-337

```python
'torch>=2.0,<3.0; sys_platform != "darwin" or platform_machine != "x86_64" or python_version < "3...
```

PyTorch stopped publishing Intel-macOS wheels after CPython 3.12.  Keep the rest of spaCR installable on that interpreter/platform pair instead of asking pip to resolve an artifact that does not exist. Apple silicon, Windows and Linux retain the normal dependency.

### lines 339-358

```python
'pyfixest>=0.40.1,<1; python_version >= "3.10"',
```

THE THREE REGRESSION BACKENDS THAT COST NOTHING TO SHIP (220). Measured 2026-08-21 rather than estimated:

pyfixest  0 new packages beyond what spaCR already pulls, >=3.10 glum      0 new packages beyond what spaCR already pulls, >=3.10 gpytorch  2 new packages, 288 KB total, >=3.10

GATED AT 3.10 BECAUSE THAT IS WHERE TORCH ALREADY SITS. `torch>=2.0` directly above resolves to 2.8.0 on Python 3.9 (the last release with cp39 wheels) and to 3.10+ builds after that, so these three add no interpreter constraint spaCR did not already have -- a 3.9 install simply does without them and still runs.

numpyro and pymer4 stay EXTRAS and that is not symmetry for its own sake. numpyro drags jax and jaxlib: 88 MB, and jaxlib requires >=3.12, so as a default it would be absent on half the interpreters spaCR supports while costing every user who never selects it. pymer4 cannot be a default at any gate -- it needs R, rpy2 and lme4, and R is a system package pip cannot provide, so `pip install pymer4` succeeds and the backend still does not run.

### lines 362-366

```python
'tensorboard>=2.21,<3.0',
```

PyTorch's official SummaryWriter backend. Vision training writes loss/accuracy/F1/LR events to each run folder for an interactive TensorBoard dashboard; 2.21 is the release exercised by the Python 3.12 event-file test, while the major-version ceiling avoids an unreviewed event-file/API break.

### lines 368-374

```python
'numba>=0.60,<1.0; sys_platform != "darwin" or platform_machine != "x86_64"',
```

SHAP 0.52 declares both numba and llvmlite without lower bounds. With numpy 2.5 that lets uv select numba 0.53.1 + llvmlite 0.36.0 from 2021: their metadata does not exclude Python 3.12, but their build scripts do, so the lightweight installer reaches a source build and fails. These floors are the first Python-3.9-through-3.12-compatible line and force the resolver onto mutually constrained numba/llvmlite wheels. Newer Python versions naturally select newer releases within these bounds.

### lines 377-390

```python
'numba>=0.60,<0.63; sys_platform == "darwin" and platform_machine == "x86_64"',
```

INTEL MAC HAS A CEILING, and it is a fact about wheels rather than about spaCR: llvmlite 0.46+ publishes no macOS x86_64 wheel, and numba 0.63+ requires that unavailable line. Without the ceiling pip selects the newest of each, finds no wheel, falls back to a source build and stops on a missing `cmake` -- reported from a clean `pip install -e .` on an iMac, where the packaged installer had succeeded minutes earlier on the same machine.

THE INSTALLER ALREADY KNEW THIS. `install_spacr_unix.sh` applies the identical pair as an architecture-specific resolver guard. Declaring it only there meant the shipped installer worked and a developer install from git did not -- the same knowledge written in one of the two places it is needed. It belongs here, where every install path reads it.

### lines 393-399

```python
'torchvision>=0.15,<1.0; sys_platform != "darwin" or platform_machine != "x86_64" or python_versi...
```

Floor RAISED from `>=0.1`, which was false — torchvision 0.1.6 is from 2017. spacr/utils.py:53 imports `ResNet18_Weights ... ResNet152_Weights` from torchvision.models.resnet at module scope (the multi-weight API, torchvision 0.13.0) and spacr/deep_spacr.py:2078 defaults to `model_name='maxvit_t'` (0.14.0). torchvision pins torch exactly, and 0.15.x is the release built against torch 2.0 — so `torch>=2.0` above already implies 0.15. This line now says that instead of implying it.

### lines 401-419

```python
'captum>=0.7.0,<1.0',
```

`torch-geometric`, `transformers`, `segmentation_models_pytorch` and `monai` REMOVED: zero imports, zero string references, zero dynamic references. Together they were the four most expensive names on the list — monai alone silently overrode the `torch>=2.0` declared two lines up (1.6.0 requires torch>=2.8.0), transformers pinned `huggingface-hub<1.0` across its entire 4.x line, and torch-geometric and smp each drag their own compiled stack — for four packages spaCR never touches. The only hit any of them produced anywhere in the tree was the word "transformers" inside a GUI description string (spacr/qt/screens/app_screen.py:195), which is prose about CNNs, not an import.

`torchcam` REMOVED from the core list -> the `attribution` extra. See blocker 2 in the numpy note above: every release satisfying `>=0.4.0,<1.0` declares `numpy<2.0.0`, so leaving it here pins numpy to 1.26.4 no matter what this file says, and on 3.13 that means a numpy source build. (0.4.1 additionally declares `requires-python >=3.11`, so in the core list it also silently backtracked to 0.4.0 on Python 3.9/3.10.)

### lines 424-434

```python
'pypdf>=6.16.1,<7.0',
```

THE SAME RULE AS THE ENTRY BELOW, applied to a module that was missing it. spacr.regression_panels imports PdfReader, PdfWriter and Link at module scope, so the PDF implementation is a direct dependency. The 6.16.1 floor is the first release containing the current resource-use fixes for hostile PDFs, including the XForm text-extraction bound; it supports every Python admitted by this package and ships a universal wheel. The discontinued PyPDF2 name is deliberately not accepted.

It also took the test suite down whole rather than one file: a module-scope import in tests/test_regression_panel_packages.py made this a COLLECTION error, so `pytest tests/` exited 2 having run nothing.

### lines 436-440

```python
'cycler>=0.10,<1',
```

DECLARED THOUGH MATPLOTLIB ALREADY PULLS IT. spacr/figure_style.py imports it directly and UNGUARDED, so a resolver that ever stops shipping it transitively turns a house-style figure into an ImportError. An import spaCR makes itself is a dependency spaCR declares itself.

### lines 443-468

```python
'PySide6>=6.6,<7',
```

THE Qt GUI IS NOT OPTIONAL. Asked for on 2026-08-17: stop hiding the GUI behind an optional extra and always install it.

`spacr-qt` is a console entry point declared UNCONDITIONALLY below, so a plain `pip install spacr` was installing a command that could not run the same shape as the pyqtgraph report earlier that day, one level up. PySide6 IS the interface: there is no second one to fall back to, so a missing binding is not a degraded install, it is no application.

SAFE ACROSS THE BLOCKING SUPPORT MATRIX, checked rather than assumed `requires-python` is >=3.9,<3.16, with 3.15 deliberately forward- declared in an experimental CI cell. A core dependency with no wheel is an install failure, not a missing feature: PySide6   6.11.2 declares <3.15,>=3.10; 6.6/6.7 cover 3.9. The `>=6.6,<7` range resolves on 3.9-3.14. The 3.15 CI cell is expected to fail until PySide6 raises its ceiling. pyqtgraph 0.14 declares >=3.10; 0.13.3 covers 3.9 and is the first release whose LabelItem base-class order works with PySide6 6.6. Same reasoning. qtawesome 1.4.2 declares >=3.7.

The HEADLESS invariant is untouched and still tested: importing spacr.core, spacr.ml or spacr.measure does not import PySide6. Being a dependency is not being an import -- a cluster install carries the wheel and never loads it.

### lines 472-474

```python
'mpmath>=1.3,<2',
```

Pure-Python arbitrary precision used by the selectable Mandelbrot backdrop.  That pattern is offered by the installed desktop app, so it must not depend on SymPy happening to bring mpmath transitively.

### lines 476-480

```python
'vispy>=0.14,<1.0',
```

The shipped backdrop defaults to the Mandelbrot deep zoom, whose shader is implemented with VisPy/gloo. VisPy is imported only when a backdrop is constructed, so declaring its wheel here makes the documented/default GUI complete without adding work to ``import spacr`` or configuration screens. Headless machines still take the guarded CPU fallback.

### lines 484-492

```python
'bottleneck>=1.3.6,<2.0',
```

KEPT despite zero imports, and deliberately so. Both are pandas' optional acceleration backends, and pandas picks them up by *presence*, never by an import spaCR could write: bottleneck accelerates the nan-aware reductions (`mean`/`sum`/`std`/`median` with skipna, which is every per-object aggregation on the measure path) and numexpr backs `pd.eval`/`DataFrame.query` and large elementwise ops. Declaring them is the only way to make the fast path reproducible rather than a property of whichever machine happens to have them. An import census will always report these as unused; that is the nature of a plugin, not a finding.

### lines 495-503

```python
'opencv-python-headless>=4.9.0.80,<5.0',
```

Both bounds are weaker than they look. The real API floor is 4.0.0 — spacr/plot.py:110 and :457 unpack `cv2.findContours` as a 2-tuple, which is the OpenCV 4 signature. 4.9.0.80 is a *wheel* floor (first tag cut after cp312 support), not an API one. `<5.0` used to be dead text — opencv-python-headless 4.12+ declares `numpy>=2`, so the old `numpy<2.0` capped the resolve at 4.11.0.86 by itself. With numpy widened, 4.12+ is now genuinely reachable and this ceiling starts doing work. OpenCV 5's Python surface was checked against all 57 cv2 symbols spaCR uses and removes none of them, so the cap is precautionary, not a known break.

### lines 505-515

```python
'pillow>=10.2.0,<13',
```

Ceiling RAISED two majors. Everything spaCR uses survives Pillow 11 and 12: `Image.Resampling.*` and the `Image.LANCZOS`-style aliases, `resize(box=, reducing_gap=)`, `ImageOps.exif_transpose`, `ImageTk`, `ImageEnhance.*`, `ImageFont.truetype`, and `PIL.ImageQt` with PySide6. The APIs Pillow 11/12 removed (PSFile, PyAccess, ImageMath.eval, isImageType, IPTC internals) appear in zero spaCR files, and the one genuine risk — `Image.fromarray(mode=)` at spacr/measure.py:2654 and spacr/deep_spacr.py:1472,1483 — is a no-op branch there, because all three pass exactly the array's own typemode ('L', 'L', 'RGB'). Verified: 12.2.0 installed, full import of every spaCR module + the attribution/packaging/smoke suites green.

### lines 520-524

```python
'readlif',
```

`aicspylibczi` removed: zero import statements and zero raw-string references anywhere under spacr/. It ships a manylinux x86_64 wheel only — no linux-aarch64, no cp313, no cp314 — and its sdist needs CMake plus libCZI headers, so it was the single dependency forcing a C source build on ARM Linux, for a package spaCR never imports.

### lines 526-530

```python
'openpyxl>=3.1,<4.0',
```

KEPT despite zero imports: it is the engine pandas requires for the two `pd.read_excel` calls at spacr/plot.py:4994 and spacr/foreign.py:968. pandas raises ImportError naming openpyxl if it is absent, and spacr/plot.py already carries that message as a string — declaring the package is what stops that message ever being shown.

### lines 533-560

```python
'umap-learn>=0.5.11,<1.0',
```

PINGOUIN IS GONE FROM THE CORE, and the cap it needed with it.

It was here because plot.py indexed pingouin's dashed result columns directly -- `.iloc[0][['T', 'p-val']]` -- which is why the ceiling had to be `<0.6`: 0.6.0 renamed every one of them ("p-val" -> "p_val"). Those two call sites were the last uses in the package, and routing plot.py's selectors through spacr.figures.stats (2026-08-17) removed them. Nothing under spacr/ imports pingouin now; the statistics it used to provide come from scipy.stats, which is already a core dependency.

It survives as a `dev` dependency, because that is what it now is: an INDEPENDENT implementation to check spaCR's own Welch ANOVA against (tests/test_spacrgraph_equal_variance.py). A wrong statistic that runs is worse than the one it replaced, so that cross-check is worth a test dependency -- and worth nothing at all in a user's install. NOT unused, whatever an import census says. spaCR reaches umap through `umap = _LazyModule('umap.umap_', block_roots=_TF_BACKED_ROOTS)` at spacr/utils.py:197 — the module name is a string literal, so there is no `import umap` anywhere in the tree for a syntax tree to find. The indirection is not incidental: importing umap eagerly costs ~6.5 s and ~1.4 GB *per worker process*, and `umap/__init__.py` pulls `parametric_umap` -> TensorFlow, which spaCR refuses. Three call sites use it: spacr/utils.py:6330 and :7019, spacr/timelapse.py:4157. 0.5.11 replaced scikit-learn's removed ``force_all_finite`` argument with ``ensure_all_finite``. Older UMAP releases fail at fit time with scikit-learn >=1.8, even though pip can otherwise resolve the pair. There was no 0.5.10 release; 0.5.11 is the first published compatible version and supports every spaCR interpreter, including Python 3.9.

### lines 562-576

```python
'xgboost>=2.0.3,<4',
```

`ttkthemes` REMOVED: zero imports. PySide6 is the interface and it carries its own theming.

Ceiling RAISED. XGBoost 3.0 removed DeviceQuantileDMatrix, `feval`, datatable support and legacy model *saving*; spaCR uses XGBClassifier (spacr/hyperparam.py:3540, spacr/regression_annotation.py:1080) plus booster-level `xgb.DMatrix`/`xgb.train`, all of which survive.

CITATIONS RE-CHECKED 2026-08-31 and three of the four were stale they pointed at ml.py and hyperparam.py lines that have since become unrelated code, and at a module that no longer exists at all. A version ceiling is only as good as the evidence for it, and evidence nobody can follow is a ceiling nobody can safely raise. The resolver handles the interpreter split by itself: xgboost 3.3 needs Python >=3.12, so a 3.9-3.11 install lands on an older compatible line without help.

### lines 578-584

```python
'biopython>=1.80,<2.0',
```

`PyWavelets` REMOVED: zero imports and zero references to `pywt`, its import name. It was pulled in as a scikit-image companion; skimage declares it itself where it needs it. `ttf_opensans` REMOVED: zero imports, and redundant besides — spaCR ships its own Open Sans under spacr/resources/font/ and loads it from there, which is why the font works today on machines that never had this package.

### lines 586-591

```python
'lxml>=5.1.0',
```

Ceiling REMOVED. spaCR imports lxml in zero files; it is here only as the preferred backend for the single `pd.read_html` call at spacr/sim.py:655, and pandas itself declares `lxml>=4.9.2` with no upper bound (and falls back to bs4+html5lib anyway). Capping a package we never import at a major we never see was constraining users' resolves for nothing. Verified: lxml 6.1.1 installed, suites green.

### lines 593-602

```python
'psutil>=5.9.8,<8',
```

Ceiling RAISED two majors. `>=5.9.8,<6.0` admitted exactly ONE release — it was an `==` pin wearing a range's clothes, and it is why the working dev env still sits on psutil 5.9.8. Nothing spaCR calls changed: psutil 6 altered `disk_partitions()` fields and `process_iter()` PID-reuse checking, and 7 removed `memory_info_ex()`; spaCR uses virtual_memory/cpu_percent/cpu_count/cpu_freq/Process/nice/cpu_affinity and already re-validates each process inside a NoSuchProcess/AccessDenied/ZombieProcess handler (spacr/utils.py:1502). `<8` rather than unbounded: psutil marks `Process.info` — used at utils.py:1508,1518 — for deprecation in 8.0.

### lines 605-624

```python
'nvidia-ml-py>=11.450.51,<14',
```

The import name is ``pynvml``, but the maintained distribution is ``nvidia-ml-py``. The separate distribution named ``pynvml`` is now a deprecated compatibility wrapper; declaring it caused both spaCR and torch.cuda imports to print a FutureWarning at startup. Function-local imports remain guarded and fall back to torch when NVML is unavailable. Pure Python, no wheel constraints; major 14 is not yet qualified.

THE FLOOR LOOKS WRONG AND IS NOT. THIS IS A DRIVER NUMBER, NOT A SEMVER. nvidia-ml-py numbers each release after the NVIDIA driver it binds, so its published sequence reads 10.418.84, 11.450.51, 11.460.79, ..., 13.610.43. The middle field is a driver branch, so 11.450.51 is NEWER than a hypothetical 11.5 and the ordinary-looking `>=11.5` names no release at all -- 11.450.51 is simply the first thing that satisfies it. DO NOT "correct" 11.450.51 to 11.5, 11.45 or 11.4.50.51. Every one of those is a bound this distribution can never equal, and the two spellings resolve identically anyway -- nothing is published between 10.418.84 and 11.450.51, so a round boundary and the exact release admit the same set. The exact release is the one that can be PINNED, which is what makes this floor a promise the `Minimum dependencies` job keeps rather than the one bound here that nothing installs.

### lines 626-633

```python
'tables>=3.8.0,<4.0',
```

`gpustat` REMOVED: zero imports. GPU state is read through GPUtil, nvidia-ml-py (imported as pynvml) and torch.cuda, all declared above. KEPT despite zero imports: PyTables is what backs `pd.HDFStore` at spacr/sequencing.py:77, which is how annotated_reads.h5 is written. The `comp_type` setting documented in spacr/settings.py:2070 is passed straight through to it as `complib`. Without this package that call is an ImportError, so it is a hard requirement that simply has no import statement — pandas owns the import.

### lines 635-653

```python
'fastremap>=1.14.1',
```

`rapidfuzz` REMOVED: zero imports. No fuzzy matching exists in spaCR; the name-matching that does exist is exact or regex. `keyring` REMOVED: zero imports. The docstring at spacr/qt/ai/__init__.py still describes an OS-keyring flow, but no code implements it — all three providers in spacr/qt/ai/providers.py shell out to the vendor CLIs (`claude`, `codex`, `gemini`) and never handle an API key. Shipping a credential-storage package for a code path that does not exist is strictly worse than not shipping it. `screeninfo` REMOVED: zero imports. It answered one question -- how big is the monitor -- for the Tkinter interface, and that interface is gone; the Qt interface asks Qt instead. Nothing else in the graph brings it in, so every headless cluster install was carrying it for nothing. Verified by walking the AST of every file under spacr/, so function-local and `try:`-guarded imports counted too, and re-checked with a raw grep for the string form the way `umap-learn` needs. KEPT despite zero imports: cellpose imports fastremap directly (as does fill_voids underneath it), and spaCR's cellpose floor is `>=4.0`. It is declared here so a cellpose that ever stops declaring it does not silently break mask relabelling.

### lines 655-662

```python
'tqdm>=4.65.0',
```

`pytz` REMOVED: zero imports. All timestamp handling is stdlib `datetime`; pandas brings its own tz support. KEPT despite zero imports: spaCR *configures* tqdm rather than calling it — spacr/cli.py:1069 sets `TQDM_DISABLE=1` when stdout is not a tty, to stop the progress bars of cellpose, captum, btrack, shap and huggingface-hub from filling a redirected log with carriage returns. That setting is only meaningful if tqdm is present, which is what this line guarantees.

### lines 664-670

```python
'requests>=2.28,<3.0',
```

ADDED. Module scope and unguarded at spacr/utils.py:2 and spacr/gui_utils.py:1, so `import spacr.utils` — i.e. importing almost anything — already required it. It was arriving only because huggingface-hub declares it, which is precisely the wrong package to rely on: hub 1.x replaces requests with httpx. `<3.0` because the `requests.HTTPError` / `requests.Timeout` names caught at spacr/utils.py:7865 are the whole API surface used.

### lines 672-675

```python
'joblib>=1.2,<2.0',
```

ADDED. `from joblib import Parallel, delayed` at module scope (spacr/utils.py:207) and function scope (spacr/object.py:49). It was arriving via scikit-learn. `>=1.2` is the release that fixed the pre-1.2 pickle deserialisation issue; nothing above it is used.

### lines 677-678

```python
'natsort>=8.0,<9.0',
```

ADDED. `from natsort import natsorted` at spacr/submodules.py:36, module scope. It was arriving via cellpose.

### lines 680-704

```python
'IPython>=8.18.1,<10',
```

`sympy` REMOVED: zero imports. `from sympy import root` at spacr/gui_elements.py:26 was the only one spaCR ever had, and that file left with the Tkinter interface. torch declares sympy for TorchDynamo and installs it regardless, so dropping the declaration costs a user nothing — it stops spaCR declaring a bound that nothing exercises. mpmath IS NOT IN THE SAME POSITION and is declared above: the Mandelbrot backdrop imports it inside four functions in spacr/qt/widgets/fractal_mandelbrot.py, and it must not depend on sympy happening to bring it transitively — which, now that sympy is only a dependency-of-a-dependency, is a weaker guarantee than it was. `wandb` REMOVED: zero imports. No experiment tracking is wired up; the run journal (spacr/run_journal.py) is spaCR's own. `openai` REMOVED: zero imports. The two textual hits are an npm one-liner (`npm install -g @openai/codex`) and a provider-name match on the string "openai" — the AI Console shells out to the `codex` CLI and never constructs an SDK client. `gdown` REMOVED: zero imports. Downloads go through requests (spacr/model_zoo.py:1290) and huggingface_hub. Ceiling RAISED. Every spaCR import is `from IPython.display import display` (plus HTML and Image in two files); all three still exist in IPython 9.x, which removed only shim modules and pre-8.16 deprecations. The cap was also redundant with the resolver: IPython 9 requires Python >=3.11, while Python 3.9 resolves the supported 8.18 line and 3.10 resolves a later 8.x on its own. Verified: IPython 9.15.0 installed, suites green.

### lines 706-708

```python
'ipywidgets>=8.1.2,<9.0',
```

`ipykernel` REMOVED: zero imports. Notebooks under Notebooks/ are run by whatever Jupyter the user already has; spaCR is not a kernel provider, and forcing a kernel into every headless cluster install bought nothing.

### lines 710-723

```python
'huggingface-hub>=0.25,<1.0',
```

`brokenaxes` REMOVED: zero imports. No broken-axis plot exists in spacr/plot.py or anywhere else. spacr only calls huggingface_hub.list_repo_files() — verified signature-identical at 0.25 and at 1.24.0, so the API is not what holds this cap down. ONE thing now does, where there used to be two: the transformers half of this note is resolved (transformers is gone, so it no longer pins `huggingface-hub<1.0` across its whole 4.x line), but hub 1.x replaced requests with httpx and `HfHubHttpError` now subclasses `httpx.HTTPError`. spacr/utils.py:7865 and its twin in spacr/gui_utils.py:1007 wrap list_repo_files in `except (requests.HTTPError, requests.Timeout)`, so on hub 1.x a network failure escapes the retry loop uncaught and reaches the user as a traceback. Widening this to `<2.0` is now a two-line source change in those two files, and nothing else.

### lines 725-732

```python
'protobuf>=5.28.3',
```

Ceiling REMOVED. spaCR imports protobuf in zero files — it is purely transitive — and this was the single tightest protobuf constraint in the entire resolved graph: google-api-core allows `<8.0.0`, onnxruntime and shap set no ceiling at all. spaCR alone was holding every user's environment at protobuf 5.x for a package it never touches. (The reverse-dependency check was re-run after `wandb` was removed above: shap and onnxruntime still require protobuf, so the floor is still buying a real thing.)

### lines 734-738

```python
]
```

win10toast lives in the Qt extras below: only spacr/qt/notify.py imports it, so a headless/core installation must not acquire a GUI notification dependency. This also keeps the core package genuinely cross-platform. 'tensorflow>=2.20.0,<3.0', 'stardist>=0.9,<1.0'

### lines 742-747

```python
name = "spacr"
```

The distribution is `spacr` (not `spacr-nightly`) so that

`pip install -e .` from a working copy replaces any prior PyPI `spacr` install instead of coexisting with it — the coexistence was the source of stale-metadata warnings after the branch rename. The `spacr-nightly` name lives on as a CLI entry-point alias below so users still have `spacr-nightly` on their PATH.

### lines 751-754

```python
name=name,
```

name/authors/urls/classifiers/requires-python are declared statically in pyproject.toml [project]; name is repeated here only because stdeb and `python setup.py egg_info` want a non-empty distribution before the pyproject config is applied.

### lines 777-781

```python
'spacr=spacr.qt:run',
```

FIVE PER-MODULE LAUNCHERS ARE GONE: `mask`, `measure`, `make_masks`, `annotate` and `classify` each opened a window of their own. Every one of them is a tab in `spacr` now, and `spacr-run <module>` is the headless route, so neither the commands nor the top-level names they occupied are needed.

### lines 785-790

```python
'spaceout=spacr.qt.spaceout:main',
```

The same application, wearing something else: rainbow palette, moving fractals behind every screen. Every module, screen and setting is the one `spacr` opens -- only the dressing differs. It is an ENTRY POINT and not a theme menu entry on purpose: nothing in Preferences offers it and nothing persists it, so an ordinary `spacr` start can never land in it.

### lines 792-796

```python
'safespacr=spacr.qt.safespacr:main',
```

The least spaCR that can still change a setting. For when a saved preference is what makes an ordinary start die: it reads every preference as its default and forces the backdrop, the setting animations, verbose logging and preloading off, so the user can get in and re-save the value that broke it.

### lines 798-801

```python
'spacr-server=spacr.qt:run_without_setup',
```

Same GUI with the first-run setup screen never offered. For a launch with nobody in front of it: the screen is modal and opens before the main window, so an unattended job would block on it.

### line 804  _(unsure)_

```python
'spacr-repro=spacr.cli_repro:main',
```

spacr-repro <run-folder> — replay a recorded run journal

### lines 806-809

```python
'spacr-workspace=spacr.cli_workspace:main',
```

spacr-workspace <run-folder> — what the run had OPEN: the databases, the montage, the figure views, and whether the files it names are still there. No display needed, because the question is usually asked about somebody else's run.

### lines 811-827

```python
'spacr-make-masks=spacr.cli_make_masks:main',
```

spacr-make-masks --folder <dir> — the mask editor opened on a folder as a RESUMABLE QUEUE (ledger 396): --order decides what is offered first, --limit ends the session, and <dir>/curate_status.csv is what makes it resume, on either machine. Importing spacr.cli_make_masks pulls no Qt, so a missing folder and an unreadable layout are both refused with a sentence over SSH rather than with a Qt crash.

DECLARED ONCE. It was declared twice on 2026-09-14 -- same name, same target, two comments -- and setuptools refuses a duplicate outright: "Duplicate element EntryPoint(name='spacr-make-masks' ...)" killed `get_requires_for_build_sdist`, so every packaging and wheel-install cell of compat-matrix went red on a build that never started. The reasoning from the second copy is kept here: the editor is mouse-driven, but choosing and ORDERING the work is not, and a curation session that cannot be pointed at a folder from a shell cannot be resumed on a second machine.

### lines 829-831

```python
'spacr-run=spacr.cli:main',
```

spacr-run <module> --settings f — headless pipeline runner for clusters: no Qt, no display. Importing spacr.cli pulls neither torch nor matplotlib, so --help/--list answer instantly.

### lines 833-838

```python
'spacr-download=spacr.cli_download:main',
```

spacr-download [what ...] — the example data as a command rather than as a button. Fetches the mask / measure / annotate example sets by default, and pieces of the 33 GB published screen only when they are asked for by kind and plate. Imports no Qt, so a cluster login node can stage the data a batch job will read.

### line 840  _(unsure)_

```python
'spacr-remote=spacr.cli_remote:main',
```

Persistent SSH / Slurm / cloud job submission and monitoring.

### line 842

```python
'spacr-plugins=spacr.cli_plugins:main',
```

Installed plugin registry and failure diagnostics.

### line 844  _(unsure)_

```python
'spacr-leakage=spacr.cli_leakage:main',
```

Standalone classifier train/test leakage audit.

### line 846  _(unsure)_

```python
'spacr-db-audit=spacr.cli_database:main',
```

SQLite health, integrity, locking, and reader/writer probe.

### lines 848-853

```python
'spacr-doctor=spacr.doctor:main',
```

Whole-installation diagnosis: which spacr is actually running (the stale-editable-install trap), which optional extras are missing, whether the GPU is usable, whether Cellpose matches the API the code calls, and whether a project database or settings file is sound. One line per check, a copyable fix on every line that is not PASS, and a non-zero exit so CI can gate on it.

### lines 858-862

```python
'dev': [
```

`tomli` on Python 3.9/3.10: tests/test_packaging_metadata.py parses pyproject.toml, and tomllib is stdlib from 3.11. The test degrades to a narrow regex without it, so this is about keeping the strong check on the oldest supported interpreter, not about being able to run the suite at all.

### lines 866-871

```python
'pytest-xdist>=3.6.1,<4',
```

CI isolates long suites by test file.  This is also the accepted way to keep the Qt suite's process-global state from accumulating over thousands of tests. Floor at the release the minimum-dependencies job installs, so the oldest xdist spaCR claims is one the Qt sharding has actually run under.

### lines 874-879

```python
'hypothesis>=6.100,<7',
```

tests/test_key_parsing_properties.py states the plate / row / column / field / object key contract as properties rather than examples -- round-trip, arity, injectivity, idempotence, and agreement between the parsers that have not been collapsed yet. It registers a `derandomize=True` profile, so it is reproducible in CI rather than a source of intermittent red.

### lines 881-884

```python
'docutils>=0.20.1,<0.24',
```

README and installer-archive rendering tests exercise the same reStructuredText parser GitHub and Sphinx use. They import docutils directly, so a contributor install must provide it instead of relying on an unrelated documentation environment.

### lines 886-890

```python
'pyarrow>=14.0.2,<26',
```

The canonical tabular-I/O contract includes Parquet and

Feather round trips. pandas deliberately leaves both engines optional; pyarrow is the one the test profile supplies. Floored at the release the minimum-dependencies job installs rather than at the start of the 14 line, which nothing runs.

### lines 892-894

```python
'ruff>=0.9,<1',
```

Quality-gate tooling. These are kept out of core dependencies: users running microscopy pipelines do not need static-analysis packages, while contributors get the same versions CI runs.

### lines 898-901

```python
'pingouin>=0.5.5,<2.0',
```

The outside reference for spaCR's own Welch ANOVA. Was a core dependency until 2026-08-17; see the note where it used to be. Unpinned at the top: the cross-check reads `F` and `p-unc`, neither of which the 0.6 column rename touched.

### lines 904-907

```python
'headless': ['opencv-python-headless>=4.9.0.80,<5.0'],
```

Pinned identically to the core dependency. Unpinned, this extra silently widened the core `<5.0` cap to "any opencv", so `pip install spacr[headless]` could resolve a different opencv than `pip install spacr`.

### lines 909-918

```python
'embeddings': ['timm>=0.9,<2.0'],
```

`pip install spacr[embeddings]` — the self-supervised feature source in `spacr/embeddings.py`. timm is the backbone registry the encoder resolves `DEFAULT_BACKBONE` through; torch is already a core dependency, so this extra is one package.

AN EXTRA RATHER THAN A CORE DEPENDENCY because the import is function-local and already refuses in a way the user can act on: `_timm_encoder` raises `EmbeddingError("self-supervised embeddings need torch and timm; install the `spacr[torch]` extra")`. Every other surface in the package imports and runs without it.

### lines 920-923

```python
'trackastra': [
```

`pip install spacr[trackastra]` — transformer-based object tracking (timelapse_mode='trackastra'). Optional because it pulls its own pretrained weights on first use; trackpy/btrack/iou stay available without it. BSD-3, PyTorch-only, no TensorFlow.

### lines 929-939

```python
'ultrack': [
```

`pip install spacr[ultrack]` — global-optimisation object tracking (timelapse_mode='ultrack'). Kept alongside trackastra rather than replacing it: ultrack solves segmentation and linking as one integer program, which wins on densely packed and 3D data, while trackastra is the better zero-config generalist. Optional because it brings an ILP solver and a database backend. BSD-3, no TensorFlow.

Ultrack 0.8 adds Python 3.13 support and retains an explicit <3.14 ceiling.  The marker makes requesting the extra on 3.14 a clean no-op instead of an unsatisfiable installation; spaCR's other tracking modes remain available there.

### lines 945-978

```python
'attribution': ['torchcam>=0.4.0,<1.0'],
```

`pip install spacr[attribution]` — the five torchcam CAM variants (gradcam, gradcam_pp, scorecam, xgradcam, layercam) in spacr/attribution.py. Optional for exactly one reason, and it is a metadata reason rather than a size one: torchcam 0.4.0 and 0.4.1 both declare `numpy<2.0.0`, which is spurious — torchcam touches numpy only in an overlay helper, and GradCAM was verified running under numpy 2.4.4 — but a declared pin is a declared pin, and in the core list it drags every install back to numpy 1.26.4, whose wheels stop at cp312. So: torchcam here, numpy 2 and Python 3.13 there. `spacr/attribution.py:580` already imports it lazily inside `_torchcam_cam`, so nothing else moves. The captum backends (integrated gradients, occlusion, ...) and spaCR's own `smoothgrad` stay available without this extra; captum is a core dependency.

KNOWN LIMITATION on Python 3.13, measured rather than inferred, and it has TWO shapes depending on how pip is invoked. Both were reproduced in a throwaway CPython 3.13.14 env on 2026-07-27:

`pip install --only-binary :all: spacr[attribution]` fails fast: ERROR: ResolutionImpossible torchcam 0.4.1 depends on numpy<2.0.0 and >=1.17.2 ... no matching distributions available for your environment: numpy a PLAIN `pip install spacr[attribution]` is worse. pip is happy to backtrack and answers "Would install ... numpy-1.26.4 torchcam-0.4.1", i.e. it drops the user into a source build of numpy 1.26.4 — the exact failure this whole change set exists to remove, and it also silently downgrades opencv to 4.11.0.86.

That is torchcam's pin to fix, not spaCR's, and it is why the extra exists rather than a wider numpy cap: the alternative was holding every spaCR user at numpy 1.26.4 and Python 3.12 to keep five CAM variants installable. On 3.13 use `spacr[qt,zernike]` and the captum backends. Upstream: frgfm/torch-cam.

### lines 980-1000

```python
'rapids': [
```

`pip install spacr[rapids]` -- GPU UMAP, t-SNE, PCA, DBSCAN and KMeans through RAPIDS cuML, with the CPU implementation kept as the default and the fallback.

AN EXTRA AND NEVER A DEPENDENCY, and the reason is Python: cuml-cu12 declares `requires_python >=3.11` and ships classifiers for 3.11 and 3.12 ONLY, while spaCR promises 3.9 through 3.14. It also wants numpy>=2.0 (no cp39 wheels) and scipy>=1.14 (needs 3.10+), so making it core would drop four of the six interpreters this project claims. The environment marker below is what keeps `pip install spacr[rapids]` from producing an unreadable resolver error on an interpreter where it cannot succeed.

Instruction 70 concluded "add neither cupy nor cucim/cuml" because it would add the rapidsai conda channel. That reason has expired RAPIDS ships PyPI wheels now -- but the conclusion stands for the stronger reason above. As an extra it constrains nothing: the base resolution is untouched on every interpreter.

NOT in the conda-forge recipe (instruction 59), which an extra makes automatic.

### lines 1005-1016

```python
'intel-gpu': [
```

`pip install spacr[intel-gpu]` — Intel Arc / Xe through torch's XPU device. NOT CORE, and not because it is large: IPEX pins a narrow torch range, so making it core would let an Intel-GPU extra dictate the torch version on the NVIDIA machines that are every current user. `spacr.accelerator` probes `torch.xpu` and simply does not select the backend when this is absent, which is the same bargain `rapids` above strikes. Instruction 319.

Linux and Windows only: there is no Intel discrete GPU on macOS, where an Intel Mac's AMD card is reached through Metal instead and Metal needs no extra at all, because torch ships it in the stock macOS wheel.

### lines 1020-1024

```python
'directml': [
```

`pip install spacr[directml]` — any vendor's GPU on Windows, via DirectML. The widest reach on that platform and the thinnest operator coverage, so it is the last backend the resolver picks. Windows-only by construction; the marker keeps it from being resolved anywhere it cannot install.

### lines 1028-1036

```python
'boosting': ['catboost>=1.2,<2.0', 'lightgbm>=4.0,<5.0'],
```

`pip install spacr[boosting]` — the two gradient-boosting backends reachable through `model_type='lightgbm'` and `model_type='catboost'` (spacr/ml.py:2477,2483 and spacr/hyperparam.py:1855,1866). Both are already imported inside the `elif` that selects them and both already raise an actionable ImportError naming the package, so an extra is what that guard was always describing. Not core because catboost alone is a ~100 MB wheel for a backend most runs never select, and because xgboost and scikit-learn's HistGradientBoosting — the two defaults — are core already.

### lines 1038-1044

```python
'plaque': ['ultralytics>=8.0,<9'],
```

`pip install spacr[plaque]` — YOLO well detection for plate or strip images in `spacr.plaque`. The import is function-local, so users who analyze one plaque field per image do not install another detection framework or download detector weights. The floor is the first Ultralytics YOLO release exposing the `YOLO(...).predict(...)` API this module calls; the major-version ceiling keeps that API contract explicit.

### lines 1046-1052

```python
'umap': ['umap-learn>=0.5.11,<1.0'],
```

`pip install spacr[umap]` — declared so the command spaCR already prints is true. `spacr/hyperparam.py:80-85` (UMAP_MISSING_MESSAGE) tells the user to run `pip install umap-learn` "or `pip install spacr[umap]`", and until now the second half named an extra that did not exist. umap-learn is also a core dependency, pinned identically, so this extra is a no-op for anyone who has spaCR installed at all — it exists to stop a printed instruction being a lie.

### lines 1054-1070

```python
'anndata': ['anndata>=0.10,<0.13'],
```

`pip install spacr[anndata]` — `spacr.anndata_export`, which writes the measurement tables out as a .h5ad so scanpy, scvi-tools and squidpy can read a spaCR run directly. Optional rather than core for one reason: anndata pulls h5py, and h5py is the one wheel in this neighbourhood that still needs a HDF5 toolchain on a platform without a prebuilt binary. Every import of it in spacr/anndata_export/ is function-local behind `require_anndata()`, which raises ANNDATA_MISSING_MESSAGE naming this extra rather than an ImportError from inside anndata's own import machinery.

The `<0.13` cap is not defensive: anndata 0.12 requires Python >=3.11, and spaCR still supports 3.10, so pip needs the room to resolve back to the 0.11 line there. scanpy is deliberately NOT declared — writing the file needs anndata alone, reading it is the user's own environment, and pinning scanpy would drag in a second copy of the leiden/igraph stack for a file spaCR only writes.

### lines 1072-1087

```python
'dinocell': ['dinocell>=0.74,<1.0'],
```

`pip install spacr[dinocell]` / `spacr[samcell]` -- the optional segmentation backends behind `segmentation_backend` (items 404 and 405). Both are imported only inside spacr/_segmentation_backends.py, whose ImportError names these extras.

KNOWN LIMITATION, read from the published metadata on 2026-09-14: dinocell 0.74 pins all 84 of its requirements to exact versions (a frozen environment rather than real bounds), among them huggingface_hub==1.7.1 against the core `huggingface-hub<1.0` cap above and numpy==2.4.3, which has no wheel before Python 3.11. So as published this extra cannot resolve alongside spaCR's core, and DINOCell needs its own environment until upstream loosens the pins. samcell 1.2.0 declares ordinary lower bounds and resolves normally.

Neither is in `all`; tests/test_packaging_metadata.py pins what `all` aggregates.

### lines 1090-1108

```python
'napari': ['napari>=0.5,<1.0'],
```

`pip install spacr[napari]` — `spacr.napari_bridge`, which hands a field's image and mask to napari, lets the user correct the mask there, and writes the corrected labels back the way spaCR writes masks with an entry in the curation ledger.

Optional rather than core, and it is not a close call: napari is a whole second application, it brings its own Qt stack next to the PySide6 spaCR's GUI already runs on, and nobody needs it to correct a mask — the Curate screen has a brush, a label picker and track curation, and records the same ledger. This extra exists for people who would rather work in the viewer they already know.

Every import of it is function-local, behind

`spacr.napari_bridge.require_napari()`, which raises NAPARI_MISSING_MESSAGE naming this extra rather than an ImportError from inside napari's own import machinery. Deliberately NOT in `all`: aggregating it would put a second GUI framework, and a second binding for the one spaCR already pins, into the extra someone types when they just want every feature.

### lines 1111-1115

```python
'sweep': ['threadpoolctl>=3.0,<4'],
```

The sweep caps BLAS threads per trial so N parallel fits do not each spawn a full thread pool -- which is instruction 114's whole subject. The import is function-local and already degrades (the limit simply is not applied), so it is an extra rather than a core dependency; scikit-learn pulls it in for most installs anyway.

### lines 1117-1121

```python
'qt': [
```

KEPT AS AN ALIAS now that these are core, so every existing instruction, the three packaging scripts, the README and the docs keep working -- `pip install spacr[qt]` resolves to the same thing `pip install spacr` does. Removing the name would break printed instructions to no benefit.

### lines 1125-1133

```python
'pyqtgraph>=0.13.3,<1',
```

THE INTERACTIVE PLOTS. The comment at the top of this file says pyqtgraph was a second, unused Qt binding pulled in by a hand-copy of cellpose's `gui` extra, and that was TRUE when it was written -- zero files imported it. It is not true any more: spacr/qt/widgets/fast_plots.py is the volcano, the Q-Q, the p-histogram, the control panel and the guide-agreement plot, and the regression results panel builds all five. Reported from a real install that had PySide6 and not this, where opening ANY module raised RuntimeError out of the screen factory.

### lines 1137-1146

```python
'flowview': ['PySide6>=6.6,<7'],
```

`pip install spacr[flowview]` — the optional live pipeline graph. Its model, tracing, layout, and static exporters stay headless-safe; the live panel reuses the PySide6 binding already pinned by core. Repeating the identical pin keeps the documented feature command valid without introducing another drawing or graph dependency.

RESTORED 2026-08-31. Dropped as collateral by 807fdd627, a commit about pandas, whose own message records that the flowview line was not its author's and had been swept in by committing a whole path. `pip install spacr[flowview]` has been an error since.

### lines 1148-1150

```python
'fractal': ['vispy>=0.14,<1.0'],
```

Backwards-compatible feature spelling. VisPy is now core because the installed application's DEFAULT_PATTERN needs it, but keeping the extra means existing ``spacr[fractal]`` commands remain valid.

### lines 1152-1154

```python
'tutorial': [
```

`spacr-tutorial` needs nothing from this extra: it prints the lesson list and opens the online tutorial library, or one lesson, in a web browser.

### lines 1159-1162

```python
'piper-tts>=1.2,<2; sys_platform != "darwin" or '
```

Piper itself has an abi3 Intel-macOS wheel, but its onnxruntime dependency has no cp314 Intel wheel. Keep the renderer available everywhere else and let 3.14 Intel Macs use the non-narrated tutorial path rather than fail the aggregate install.

### lines 1166-1170

```python
'czi': ['pylibCZIrw>=5.0.0,<7.0; python_version < "3.14"',
```

The AI Console shells out to vendor coding-agent CLIs

(`claude`, `codex`, `gemini`) so authentication piggy-backs on the user's chat subscription — no Python API SDKs needed. Users install whichever CLI(s) they want separately; see the Providers… dialog in the AI Console for one-liners.

### lines 1172-1196

```python
'czi': ['pylibCZIrw>=5.0.0,<7.0; python_version < "3.14"',
```

File-format extras.

These name readers for vendor microscope formats. pylibCZIrw is intentionally optional and imported only when its streaming CZI converter is selected: it does not yet publish a CPython 3.14 wheel. The pure-Python czifile reader remains in core, so CZI data is still readable on 3.14.

Which of these actually gate the platform matrix (re-verified against the PyPI JSON API on 2026-07-27, not assumed): mahotas 1.4.18 — cp310, cp311, cp312 and nothing else, with manylinux x86_64 Linux wheels ONLY. No cp313 or cp314 has ever been published at any version, and the last aarch64 build was 1.4.13. This is the pin that used to cap Python at 3.12 next to numpy and force a C++ build on ARM Linux; moving it here is what lets `requires-python` reach <3.14. pylibCZIrw — NOT a 3.13 blocker, contrary to earlier analysis. 5.1.1 (inside the `<6.0` cap already declared above) ships cp39-cp313 including manylinux aarch64 and macosx arm64. It is only a 3.14 blocker: no cp314 exists even in 6.1.0. czifile, readlif, nd2reader — all pure `py3-none-any`. They constrain no platform and no Python version; these extras are organisational, not load-bearing.

### lines 1202-1204

```python
'btrack': ['btrack>=0.7.0,<1.0'],
```

btrack loads a native tracker library. Keeping it behind a lazy feature boundary lets the rest of timelapse run on new Python versions while upstream wheels catch up.

### lines 1207-1229

```python
'numpyro': ['numpyro>=0.13,<1.0', 'jax>=0.4,<1.0'],
```

`pip install spacr[numpyro]` / `spacr[pymc]` — the two exact-NUTS backends of `spacr.power_model`, which fits the horseshoe Poisson hit model. Both are already imported inside the branch that selects them (`_fit_numpyro_nuts`, `_fit_pymc_nuts`) and `resolve_backend` already refuses to substitute one for another, naming the missing package — so an extra is what that guard was always describing, the same reasoning as `boosting`.

Two extras rather than one because they are ALTERNATIVES, not a pair: the model needs one exact sampler, and `backend="auto"` takes numpyro if it is there, else pymc, else the torch ADVI that is always available. Making one extra install both would force a second multi-hundred-megabyte inference stack on a user who has already chosen the other. jax is named alongside numpyro because `_fit_numpyro_nuts` imports `jax` and `jax.numpy` directly rather than only through numpyro, and the dependency census — correctly — counts an import it can see.

Neither is in `all`, for the reason the exclusions below give: jax resolves to a platform-specific build (CPU/CUDA/ROCm/Metal) and pymc brings PyTensor and a C compiler path. Someone typing `all` wants every feature, not a second numerical-computing runtime chosen for them.

### lines 1232-1242

```python
'pyfixest': ['pyfixest>=0.40.1,<1; python_version >= "3.10"'],
```

Optional regression engines selected by regression_backend. Both imports are function-local, and the GUI disables an engine that is not installed, so neither belongs in the core scientific stack. These floors are the oldest Python 3.9 wheels checked against the exact APIs spaCR calls: pyfixest.core.demean.demean and glum's GeneralizedLinearRegressor offset/sample_weight fit path. CORE NOW (220), and these names stay as aliases so a script that says `pip install spacr[pyfixest]` keeps working. The pin must be BYTE-IDENTICAL to the core one -- there is a test asserting an extra never contradicts core, because two pins for one package is how an environment ends up with a version neither line asked for.

### lines 1247-1268

```python
'zarr': ['zarr>=2.16,<4', 'numcodecs>=0.12,<1'],
```

`pip install spacr[zarr]` — `spacr.ome_zarr`, the OME-NGFF

(OME-Zarr) reader/writer. The emerging standard for large bioimaging data: chunked, so a 100 GB plate is readable a tile at a time instead of all at once, and multiscale, so a plate overview does not decode full resolution to draw 200 px.

It is an extra rather than a core dependency, and the reason is narrow: spaCR parses and writes the OME-NGFF *metadata* itself (`multiscales`, `axes`, `coordinateTransformations`) in pure Python, because that layout is the thing worth getting right and a library would only hide it. What zarr/numcodecs are needed for is the chunk CODEC — blosc, zstd, lz4 — which is where the compiled code lives. `spacr.ome_zarr` reads and writes stored/zlib chunks with the standard library alone, so a spaCR-written OME-Zarr round-trips on a plain `pip install spacr`; anything compressed with a third-party codec raises a message naming this extra rather than a traceback.

`zarr>=2.16,<4` deliberately spans the v2/v3 rewrite: v2 reads zarr-format 2 (which is what OME-NGFF 0.4 is), v3 reads both, and `spacr.ome_zarr` supports either at runtime. numcodecs is named explicitly rather than relied on through zarr, because the codec is what is actually imported.

### lines 1270-1283

```python
'omero': ['omero-py>=5.17,<6'],
```

`pip install spacr[omero]` — `spacr.omero`, importing a dataset or plate by id from an OMERO server and exporting spaCR results back as annotations.

Deliberately NOT in `all`, and this is the one exclusion that is not about wheels being missing: omero-py depends on zeroc-ice, a compiled C++ Ice runtime whose wheels lag Python releases by a long way and which otherwise needs a C++ toolchain plus the Ice development headers. Putting it in the extra most likely to be typed by someone who just wants everything would turn `pip install spacr[all]` into a source build of a C++ middleware stack, which is the same class of failure `attribution` is excluded for. Anyone who has an OMERO server has an installed Ice already, or knows they need one; nobody else should pay for it.

### lines 1286-1346

```python
'all': [
```

`pip install spacr[all]` — every optional feature at once, minus ten, each for a stated reason: `dev`   — test tooling, not a feature. `full`  — the GUI-capable opencv build, which would shadow the headless one already in the core deps. `umap`  — a pure alias for a core dependency; aggregating it would add nothing and only make the union harder to read. `attribution` — DELIBERATE, and the one that is not obvious. torchcam declares `numpy<2.0.0`, which has no cp313 wheel. Including it here would mean that on Python 3.13 `pip install spacr[all]` backtracks into a SOURCE BUILD of numpy 1.26.4 (measured: pip reports "Would install ... numpy-1.26.4 torchcam-0.4.1" and silently downgrades opencv to 4.11.0.86) — the exact failure this change set out to remove, reintroduced through the extra most likely to be typed by someone who just wants everything. `spacr[all,attribution]` remains available on 3.9-3.12 for anyone who wants both. `omero` — same class of exclusion as `attribution`, for the reason spelled out at its own entry: omero-py pulls zeroc-ice, a compiled C++ Ice runtime, and `all` must not turn into a middleware source build. `numpyro`, `pymc` — the two exact-NUTS backends are alternatives to each other and to a torch path that is always present, so `all` would install two inference stacks to use at most one. jax also resolves to a platform-specific build and pymc brings PyTensor's compiler path; neither belongs in the extra someone types when they just want every feature. `pyfixest`, `glum` — optional alternatives to the statsmodels regression engine already in the core stack. They are selected explicitly per fit, so installing both for every `all` user would add weight without making another module available. `zarr`  — kept out because it buys nothing for a user who did not ask for it: `spacr.ome_zarr` reads and writes stored/zlib OME-Zarr with the standard library alone, and numcodecs (the part that is actually compiled) only matters for third-party chunk codecs. Anyone who has blosc-compressed NGFF data knows they do.

Spelled out as concrete requirements rather than a recursive

`spacr[qt,tutorial,...]` self-reference so it resolves identically on old pip, under `pip download`, and in uv/conda resolvers. tests/test_packaging_metadata.py asserts this stays the exact union of the extras it claims to aggregate, so it cannot drift.

Ultrack 0.8 now supports Python 3.13 and declares <3.14. Trackastra declares >=3.10. Their markers below keep ``all`` resolvable at both ends of spaCR's 3.9--3.14 range while installing each tracker where upstream actually supports it.

`spacr[all]` also pulls mahotas, which has no cp313 wheel and builds from sdist. That one is deliberate and fine — asking for *everything* may reasonably require a toolchain, and unlike torchcam it succeeds where one exists (verified: mahotas 1.4.18 built and ran against numpy 2.4.4 on 3.13). A plain `pip install spacr` never needs a compiler, which is the whole point of the split.

### lines 1350-1354

```python
'pyqtgraph>=0.13.3,<1',
```

The interactive plots. In `qt` since the regression panel was built on them, and missing here until 2026-08-17 -- so `pip install spacr[all]` installed strictly LESS than `pip install spacr[qt]`, which is the one thing the name of this extra promises cannot happen.
