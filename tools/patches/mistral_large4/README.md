# Mistral Large 4 startup dependency patches

These temporary patches accompany the Ascend MLA cache and DSpARK startup
fixes. Apply them to the corresponding dependencies during environment setup.
The Python wheel does not apply these files automatically.

## Tested dependencies

| Patch | Dependency | Failure addressed |
| ----- | ---------- | ----------------- |
| `vllm-0.34.0-oot-import.patch` | vLLM 0.34.0, commit `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` | Importing the AMD implementation adapted by an out-of-tree platform fails in the parent package's platform check. |
| `cann-9.3.0-metadata-device.patch` | CANN 9.3.0 `cann_ops_transformer` | QLI and FlashMLA allocate too little metadata when Torch's device properties differ from ACL's native core counts. |

Check each patch against the installed dependency before applying it. Other
dependency revisions require a fresh compatibility check. Once upstream vLLM
accepts out-of-tree imports and CANN uses native per-device metadata capacities,
use the fixed dependencies and remove the corresponding patch.

## Apply

From this vLLM Ascend checkout, record the patch directory:

```bash
PATCH_DIR="$(pwd)/tools/patches/mistral_large4"
```

Apply the vLLM change in the source checkout used by the serving environment:

```bash
git -C /path/to/vllm apply --check "$PATCH_DIR/vllm-0.34.0-oot-import.patch"
git -C /path/to/vllm apply "$PATCH_DIR/vllm-0.34.0-oot-import.patch"
```

Locate CANN through the serving environment's Python, then apply the SDK patch
relative to its package directory:

```bash
CANN_SITE_PACKAGES=$(python -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("cann_ops_transformer").origin).resolve().parent.parent)')
git -C "$CANN_SITE_PACKAGES" apply --check "$PATCH_DIR/cann-9.3.0-metadata-device.patch"
git -C "$CANN_SITE_PACKAGES" apply "$PATCH_DIR/cann-9.3.0-metadata-device.patch"
python -m pytest -q "$PATCH_DIR/test_metadata_device.py"
```

Apply the same changes in every worker's serving environment before launching
the service. Persist the SDK change in the image or repeat environment setup
when recreating a container. Review a failed `--check` against the dependency's
source, including whether the fix is already present.

## Scope and behavior

The vLLM patch lets any out-of-tree platform explicitly import the backend it
adapts. CUDA and ROCm keep their existing exports; unsupported in-tree platforms
keep the existing error. It adds no automatic backend selection for plugins.

The CANN patch uses `acl.rt.get_device_info` for AICore and Vector counts, keyed
by the tensor's device. Both metadata frontends share the query helper.
Successful queries are cached per device for the process lifetime; failed
queries raise and are retried on the next call. Torch continues to own runtime
initialization. There is no model-name check, chip-name check, capacity floor,
or fallback count. Recreate the worker process after changing device visibility.

The SDK tests cover devices with different capacities, smaller and larger
capacities, cache isolation, failed queries, recovery after failure, and invalid
devices. They require the patched SDK and mock the native calls; they do not
allocate NPU tensors or establish kernel correctness.

## Observed validation

On two Ascend 950DT hosts with eight NPUs each, the installed QLI and FlashMLA
metadata interfaces passed for batch sizes 1 and 33 on all 16 devices, totaling
64 native calls. A device reporting 36/72 native AICore/Vector counts used those
counts; devices reporting 32/64 used their own counts.

With the Ascend cache/DSpARK fixes and both dependency patches installed, native
FP8 Mistral Large 4 completed graph capture on all 16 workers using TP8, DP2,
EP16, DSpARK with three draft tokens, and shared-expert DP. Health and model
listing returned HTTP 200; two concurrent generation requests ended naturally.
The server advertised a 128000-token context limit. This startup smoke does not
establish full-length context correctness, model accuracy, or performance.
