import torch

from FastBlend.src.backends import taichi_backend
from FastBlend.src.patch_match import taichi_available


def test_parity():
    if not taichi_available:
        print("Taichi not available, skipping parity test.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Testing on device: {device}")

    B, H, W, C = 1, 64, 64, 3
    patch_size = 7
    pad_size = patch_size // 2

    # Generate random data
    torch.manual_seed(42)
    source = torch.rand((B, H + 2 * pad_size, W + 2 * pad_size, C), device=device)
    target = torch.rand((B, H + 2 * pad_size, W + 2 * pad_size, C), device=device)

    # Identity NNF [y, x]
    y_coords = torch.arange(H, device=device).view(H, 1).repeat(1, W)
    x_coords = torch.arange(W, device=device).view(1, W).repeat(H, 1)
    nnf = (
        torch.stack([y_coords, x_coords], dim=2)
        .unsqueeze(0)
        .repeat(B, 1, 1, 1)
        .to(torch.int32)
    )

    # Create Taichi matcher
    matcher_taichi = taichi_backend.PatchMatcherTaichi(H, W, C, patch_size, num_iter=1)

    # 1. Test remap
    print("Testing remap...")
    remapped_taichi = matcher_taichi.apply_nnf_to_image(nnf, source)

    # Reference remap (Identity NNF should result in source)
    # But wait, FastBlend remap is a voting-based reconstruction.
    # If NNF is identity, for a target pixel (y, x), all covering patches (py, px)
    # will point to source(match_Y - py, match_X - px) = (y - py, x - px).
    # Since match is identity, (y, x) covered by patch centered at (y+py, x+px)
    # where match is (y+py, x+px), so it points to source((y+py)-py, (x+px)-px) = (y, x).
    # So all contributions should be source[y, x].

    source_unpadded = source[:, pad_size:-pad_size, pad_size:-pad_size, :]
    remapped_taichi_unpadded = matcher_taichi.unpad_image(remapped_taichi)

    diff_remap = (remapped_taichi_unpadded - source_unpadded).abs().max()
    print(f"Remap Max Diff: {diff_remap.item()}")
    if diff_remap > 1e-5:
        print("FAILED: Remap diff too large!")
    else:
        print("SUCCESS: Remap matches identity.")

    # 2. Test patch_error
    print("Testing patch_error...")
    error_taichi = matcher_taichi.get_patch_error(source, nnf, target)

    # Reference patch_error
    # SSD: sum((target[y+py, x+px] - source[match_Y+py, match_X+px])**2)
    def ref_ssd(s, n, t, p, pad):
        B, H, W, _ = n.shape
        r = p // 2
        err = torch.zeros((B, H, W), device=s.device)
        for b in range(B):
            for y in range(H):
                for x in range(W):
                    sy = n[b, y, x, 0]
                    sx = n[b, y, x, 1]
                    # Target patch
                    tp = t[
                        b, y + pad - r : y + pad + r + 1, x + pad - r : x + pad + r + 1
                    ]
                    # Source patch
                    sp = s[
                        b,
                        sy + pad - r : sy + pad + r + 1,
                        sx + pad - r : sx + pad + r + 1,
                    ]
                    err[b, y, x] = ((tp - sp) ** 2).sum()
        return err

    error_ref = ref_ssd(source, nnf, target, patch_size, pad_size)
    diff_error = (error_taichi - error_ref).abs().max()
    print(f"Error Max Diff: {diff_error.item()}")
    if diff_error > 1e-4:
        print("FAILED: Error diff too large!")
    else:
        print("SUCCESS: Error matches reference.")

    # 3. Test iteration (propagation/search)
    print("Testing estimation...")
    # This might have randomness, but can check for stability
    nnf_new, style_new = matcher_taichi.estimate_nnf(source, target, source, nnf)
    print("Estimation complete.")


if __name__ == "__main__":
    test_parity()
