# comfyui_ezsynth/nodes/base.py
"""
Base class for ReEzSynth ComfyUI nodes.
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch


class EZBaseNode:
    """
    Base class for all ReEzSynth ComfyUI nodes.

    This class provides common functionality and enforces
    the required interface for ComfyUI nodes.
    """

    CATEGORY = "ReEzSynth"

    RETURN_TYPES: tuple = ()
    RETURN_NAMES: tuple = ()
    FUNCTION: str = "execute"
    OUTPUT_NODE: bool = False

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Define the input types for this node.

        Returns:
            Dictionary defining required and optional inputs
        """
        return {"required": {}}

    def _format_output(
        self, output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Format output tensor for ComfyUI.
        Ensures all tensors have 3 channels (expands single-channel to RGB).

        Args:
            output: Output tensor(s)

        Returns:
            Formatted output tensor(s) with 3 channels
        """
        if isinstance(output, tuple):
            return tuple(self._format_single_tensor(t) for t in output)
        return self._format_single_tensor(output)

    def _format_single_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Format a single tensor for ComfyUI output.
        Ensures the tensor has 3 channels.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with 3 channels
        """
        print(f"DEBUG _format_single_tensor: input shape {tensor.shape}")
        
        # Handle tuple output format (B, H, W, C)
        if tensor.dim() == 4:
            if tensor.shape[-1] == 1:
                # Single channel: expand to 3 channels
                tensor = tensor.repeat(1, 1, 1, 3)
                print(f"DEBUG _format_single_tensor: expanded 4D single channel to RGB: {tensor.shape}")
        elif tensor.dim() == 3:
            if tensor.shape[-1] == 1:
                # Single channel: expand to 3 channels
                tensor = tensor.repeat(1, 1, 3)
                print(f"DEBUG _format_single_tensor: expanded 3D single channel to RGB: {tensor.shape}")
        
        print(f"DEBUG _format_single_tensor: output shape {tensor.shape}")
        return tensor

    def _validate_image_input(
        self, image: torch.Tensor, param_name: str = "image"
    ) -> torch.Tensor:
        """
        Validate and normalize image input tensor.

        Args:
            image: Input image tensor
            param_name: Name of the parameter for error messages

        Returns:
            Validated image tensor
        """
        if image is None:
            raise ValueError(f"{param_name} cannot be None")

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"{param_name} must be a torch.Tensor, got {type(image)}"
            )

        # Ensure tensor is on CPU for image operations
        if image.is_cuda:
            image = image.cpu()

        # Handle ComfyUI format (B, H, W, C) or (H, W, C)
        if image.dim() == 4:
            # Take first image from batch if batch dimension exists
            image = image[0]

        return image

    def _validate_guide(
        self, guide: torch.Tensor, weight: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """
        Validate guide input.

        Args:
            guide: Guide image tensor
            weight: Guide weight

        Returns:
            Tuple of (validated_guide, weight)
        """
        guide = self._validate_image_input(guide, "guide")
        return guide, weight

    def execute(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute the node's logic.

        Args:
            *args: Positional arguments from input sockets
            **kwargs: Keyword arguments from input sockets

        Returns:
            Output to be sent to output sockets
        """
        raise NotImplementedError("Subclasses must implement execute()")
