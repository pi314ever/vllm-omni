# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]
from vllm.logger import init_logger

# #region agent log
from vllm_omni.diffusion.debug_mem_logger import log_event
from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

# #endregion

logger = init_logger(__name__)


class SequentialOffloadHook(ModelHook):
    """Hook for sequential offloading with mutual exclusion on encoder and DiT modules.

    To be used as a model-level (or "component-level") of CPU offloading method;
    When a module's forward is called, this hook offloads target modules to CPU
    and loads the current module to GPU.
    """

    _HOOK_NAME = "sequential_offload"

    def __init__(
        self,
        offload_targets: list[nn.Module],
        device: torch.device,
        pin_memory: bool = True,
        use_hsdp: bool = False,
    ):
        # Modules to offload to CPU before this module runs
        self.offload_targets = offload_targets
        self.device = device
        self.pin_memory = pin_memory
        self.use_hsdp = use_hsdp

    @staticmethod
    def _move_params(
        module: nn.Module,
        target_device: torch.device,
        *,
        non_blocking: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """Move module parameters and buffers to device.

        This cls method specifically prevents recursion device movement,
        E.g., Cache-DiT CachedBlocks has attr `transformer` as a ref to original
        transformer blocks, thus `module.to(device)` will fail for recursion calling,
        refer to
        https://github.com/vipshop/cache-dit/blob/v1.2.3/src/cache_dit/caching/cache_blocks/__init__.py#L83
        """
        for p in module.parameters():
            if p.data.device != target_device:
                data = p.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                p.data = data
        for b in module.buffers():
            if b.device != target_device:
                data = b.data.to(target_device, non_blocking=non_blocking)
                if pin_memory and target_device.type == "cpu" and not isinstance(data, DTensor):
                    data = data.pin_memory()
                b.data = data

    def _to_cpu(self, module: nn.Module) -> None:
        try:
            param = next(module.parameters())
        except StopIteration:
            return

        if param.device.type == "cpu":
            return

        self._move_params(
            module,
            torch.device("cpu"),
            non_blocking=not self.use_hsdp,
            pin_memory=self.pin_memory,
        )
        current_omni_platform.empty_cache()

    def _to_gpu(self, module: nn.Module) -> None:
        try:
            if next(module.parameters()).device == self.device:
                return
        except StopIteration:
            return

        # #region agent log
        _gpu_method = "bulk_to"
        # #endregion
        try:
            module.to(self.device)
        except (RecursionError, RuntimeError):
            # #region agent log
            _gpu_method = "move_params_fallback"
            # #endregion
            self._move_params(module, self.device, non_blocking=False)
        # #region agent log
        log_event(
            "sequential_backend.py:_to_gpu:done",
            f"module moved to GPU via {_gpu_method}",
            data={
                "module": module.__class__.__name__,
                "method": _gpu_method,
            },
            hypothesis_id="H_OOM_FIX",
        )
        # #endregion

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        # #region agent log
        log_event(
            "sequential_backend.py:pre_forward:before_offload",
            "offload hook: before CPU/GPU swap",
            data={
                "module": module.__class__.__name__,
                "offload_targets": [t.__class__.__name__ for t in self.offload_targets],
            },
            hypothesis_id="OFFLOAD",
        )
        # #endregion
        # Offload target modules to CPU
        for target in self.offload_targets:
            self._to_cpu(target)

        # Load current module to GPU
        self._to_gpu(module)
        current_omni_platform.synchronize()

        # #region agent log
        log_event(
            "sequential_backend.py:pre_forward:after_offload",
            "offload hook: after CPU/GPU swap",
            data={
                "module_on_device": module.__class__.__name__,
            },
            hypothesis_id="OFFLOAD",
        )
        # #endregion

        logger.debug(
            "Swapped: %s -> CPU, %s -> %s, free memory: %.4f GB",
            [t.__class__.__name__ for t in self.offload_targets],
            module.__class__.__name__,
            f"{self.device.type}:{self.device.index}",
            current_omni_platform.get_free_memory() / 1024 / 1024 / 1024,
        )

        return args, kwargs


def apply_sequential_offload(
    all_modules: list[nn.Module],
    device: torch.device,
    pin_memory: bool = True,
    use_hsdp: bool = False,
) -> None:
    """Apply sequential offloading hooks with full mutual exclusion.

    Each module offloads ALL other modules to CPU before loading itself
    to GPU. This ensures only one component occupies GPU memory at a time.

    Args:
        all_modules: All pipeline modules to participate in offloading
        device: Target GPU device for loading
        pin_memory: Whether to pin CPU memory for faster transfers
        use_hsdp: Whether HSDP is enabled (affects non_blocking behavior)

    Example:
        >>> apply_sequential_offload(
        ...     dit_modules=[pipeline.transformer],
        ...     encoder_modules=[pipeline.text_encoder, pipeline.vae],
        ...     device=torch.device("cuda:0"),
        ... )
        >>> # Modules of pipeline now automatically swap between CPU and GPU
    """
    for module in all_modules:
        offload_targets = [m for m in all_modules if m is not module]
        registry = HookRegistry.get_or_create(module)
        hook = SequentialOffloadHook(
            offload_targets=offload_targets,
            device=device,
            pin_memory=pin_memory,
            use_hsdp=use_hsdp,
        )
        registry.register_hook(SequentialOffloadHook._HOOK_NAME, hook)
        logger.debug(
            "Registered offload hook for %s (targets: %d others)", module.__class__.__name__, len(offload_targets)
        )


def remove_sequential_offload(modules: list[nn.Module]) -> None:
    """Remove sequential offloading hooks from modules.

    Args:
        modules: Modules to remove hooks from

    Example:
        >>> all_modules = [*dit_modules, *encoder_modules]
        >>> remove_sequential_offload(all_modules)
    """
    for module in modules:
        registry: HookRegistry | None = getattr(module, "_hook_registry", None)
        if registry is not None:
            registry.remove_hook(SequentialOffloadHook._HOOK_NAME)
            logger.debug("Removed offload hook from %s", module.__class__.__name__)


class ModelLevelOffloadBackend(OffloadBackend):
    """Model-level (sequential) offloading backend.

    Uses SequentialOffloadHook registered via HookRegistry for automatic module swapping.
    """

    def __init__(self, config: OffloadConfig, device: torch.device):
        super().__init__(config, device)
        self._offload_modules: list[nn.Module] = []  # Track modules with hooks

    def enable(self, pipeline: nn.Module) -> None:
        if self.enabled:
            logger.warning("ModelLevelOffloadBackend already enabled")
            return

        # #region agent log
        log_event(
            "sequential_backend.py:enable:start", "ModelLevelOffloadBackend.enable() called", hypothesis_id="OFFLOAD"
        )
        # #endregion

        modules = ModuleDiscovery.discover(pipeline)
        if not modules.dits:
            logger.warning("No DiT/transformer modules found, skipping model-level offloading")
            return
        if not modules.encoders:
            logger.warning("No encoder modules found, skipping model-level offloading")
            return

        all_modules: list[nn.Module] = []
        all_names: list[str] = []

        all_modules.extend(modules.dits)
        all_names.extend(modules.dit_names)
        all_modules.extend(modules.encoders)
        all_names.extend(modules.encoder_names)
        all_modules.extend(modules.auxiliaries)
        all_names.extend(modules.auxiliary_names)
        all_modules.extend(modules.vaes)
        all_names.extend(modules.vae_names)

        # #region agent log
        _all_attrs = [
            a for a in dir(pipeline) if not a.startswith("_") and isinstance(getattr(pipeline, a, None), nn.Module)
        ]
        log_event(
            "sequential_backend.py:enable:modules_discovered",
            "modules discovered",
            data={
                "dit_names": modules.dit_names,
                "encoder_names": modules.encoder_names,
                "auxiliary_names": modules.auxiliary_names,
                "vae_found": bool(modules.vaes),
                "all_managed": all_names,
                "all_module_attrs": _all_attrs,
                "undiscovered_modules": [a for a in _all_attrs if a not in all_names],
            },
            hypothesis_id="OFFLOAD",
        )
        # #endregion

        # Move ALL modules to CPU so only the active one occupies GPU
        for mod in all_modules:
            try:
                SequentialOffloadHook._move_params(mod, torch.device("cpu"))
            except Exception as exc:
                logger.debug("Failed to move %s to CPU: %s", mod.__class__.__name__, exc)
        current_omni_platform.empty_cache()

        apply_sequential_offload(
            all_modules=all_modules,
            device=self.device,
            pin_memory=self.config.pin_cpu_memory,
            use_hsdp=self.config.use_hsdp,
        )

        self._offload_modules = all_modules
        self.enabled = True

        # #region agent log
        log_event(
            "sequential_backend.py:enable:done",
            "offload hooks applied, all modules on CPU",
            data={
                "all_managed": all_names,
            },
            hypothesis_id="OFFLOAD",
        )
        # #endregion

        logger.info(
            "Model-level offloading enabled: %s (full mutual exclusion)",
            ", ".join(all_names),
        )

    def disable(self) -> None:
        if not self.enabled:
            return

        remove_sequential_offload(self._offload_modules)

        self._offload_modules.clear()
        self.enabled = False
        logger.info("Model-level offloading disabled")
