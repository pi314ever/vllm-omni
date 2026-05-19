# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn
from torch.distributed._tensor import DTensor  # type: ignore[attr-defined]
from vllm.logger import init_logger

from vllm_omni.diffusion.hooks import HookRegistry, ModelHook
from vllm_omni.platforms import current_omni_platform

from .base import OffloadBackend, OffloadConfig
from .module_collector import ModuleDiscovery

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

        self._move_params(module, self.device, non_blocking=False)

    def pre_forward(self, module: nn.Module, *args, **kwargs) -> tuple[tuple, dict]:
        # Offload target modules to CPU
        for target in self.offload_targets:
            self._to_cpu(target)

        # Load current module to GPU
        self._to_gpu(module)
        current_omni_platform.synchronize()

        logger.debug(
            "Swapped: %s -> CPU, %s -> %s, free memory: %.4f GB",
            [t.__class__.__name__ for t in self.offload_targets],
            module.__class__.__name__,
            f"{self.device.type}:{self.device.index}",
            current_omni_platform.get_free_memory() / 1024 / 1024 / 1024,
        )

        return args, kwargs


_HOOKABLE_VAE_METHODS = ("decode", "encode")


def apply_sequential_offload(
    all_modules: list[nn.Module],
    device: torch.device,
    pin_memory: bool = True,
    use_hsdp: bool = False,
    vae_modules: list[nn.Module] | None = None,
) -> None:
    """Apply sequential offloading hooks with full mutual exclusion.

    Each module offloads ALL other modules to CPU before loading itself
    to GPU. This ensures only one component occupies GPU memory at a time.

    Args:
        all_modules: All pipeline modules to participate in offloading
        device: Target GPU device for loading
        pin_memory: Whether to pin CPU memory for faster transfers
        use_hsdp: Whether HSDP is enabled (affects non_blocking behavior)
        vae_modules: VAE modules whose decode/encode methods should also
            be wrapped so that offload hooks fire for those calls

    Example:
        >>> apply_sequential_offload(
        ...     all_modules=[pipeline.transformer, pipeline.text_encoder, pipeline.vae],
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

    # Wrap decode/encode on VAE modules so offload hooks fire
    for vae_mod in vae_modules or []:
        registry = HookRegistry.get_or_create(vae_mod)
        for method_name in _HOOKABLE_VAE_METHODS:
            if hasattr(vae_mod, method_name):
                registry.wrap_method(method_name)
                logger.debug(
                    "Wrapped %s.%s for offload hook dispatch",
                    vae_mod.__class__.__name__,
                    method_name,
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
            registry.unwrap_all_methods()
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
            vae_modules=modules.vaes,
        )

        self._offload_modules = all_modules
        self.enabled = True

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
