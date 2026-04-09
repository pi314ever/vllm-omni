# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class PipelineModules:
    dits: list[nn.Module]
    dit_names: list[str]
    encoders: list[nn.Module]
    encoder_names: list[str]
    auxiliaries: list[nn.Module]
    auxiliary_names: list[str]
    vaes: list[nn.Module]
    vae_names: list[str]


class ModuleDiscovery:
    """Discovers pipeline components for offloading"""

    DIT_ATTRS = ["transformer", "transformer_2", "dit", "sr_dit", "language_model", "transformer_blocks", "model"]
    ENCODER_ATTRS = ["text_encoder", "text_encoder_2", "text_encoder_3", "image_encoder"]
    AUXILIARY_ATTRS = ["connectors", "vocoder"]
    VAE_ATTRS = ["vae", "audio_vae"]

    @staticmethod
    def discover(pipeline: nn.Module) -> PipelineModules:
        """Discover DiT, encoder, and VAE modules from pipeline.

        Args:
            pipeline: Diffusion pipeline model

        Returns:
            PipelineModules with lists of discovered modules and names
        """
        # Collect DiT/transformer modules
        dit_modules: list[nn.Module] = []
        dit_names: list[str] = []
        for attr in ModuleDiscovery.DIT_ATTRS:
            if not hasattr(pipeline, attr):
                continue
            module_obj = getattr(pipeline, attr)
            if module_obj is None:
                continue

            if not isinstance(module_obj, nn.Module):
                logger.warning(f"Expected {attr} to be nn.Module, got {type(module_obj)!r}")
                continue

            if module_obj in dit_modules:
                continue

            dit_modules.append(module_obj)
            dit_names.append(attr)

        # Collect all encoders
        encoders: list[nn.Module] = []
        encoder_names: list[str] = []
        for attr in ModuleDiscovery.ENCODER_ATTRS:
            if hasattr(pipeline, attr) and getattr(pipeline, attr) is not None:
                encoders.append(getattr(pipeline, attr))
                encoder_names.append(attr)

        # Collect auxiliary modules (connectors, vocoder, etc.)
        auxiliaries: list[nn.Module] = []
        auxiliary_names: list[str] = []
        for attr in ModuleDiscovery.AUXILIARY_ATTRS:
            module = getattr(pipeline, attr, None)
            if module is not None and isinstance(module, nn.Module):
                auxiliaries.append(module)
                auxiliary_names.append(attr)

        # Collect all VAE submodules (e.g. video VAE + audio VAE)
        vaes: list[nn.Module] = []
        vae_names: list[str] = []
        for attr in ModuleDiscovery.VAE_ATTRS:
            module = getattr(pipeline, attr, None)
            if module is not None and isinstance(module, nn.Module):
                vaes.append(module)
                vae_names.append(attr)

        return PipelineModules(
            dits=dit_modules,
            dit_names=dit_names,
            encoders=encoders,
            encoder_names=encoder_names,
            auxiliaries=auxiliaries,
            auxiliary_names=auxiliary_names,
            vaes=vaes,
            vae_names=vae_names,
        )
