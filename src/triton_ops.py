from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _rmsnorm_fwd_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        stride_x_row,
        stride_out_row,
        N,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_ptrs = x_ptr + row_idx * stride_x_row + cols
        x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        w_ptrs = w_ptr + cols
        w = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / N
        rsqrt = 1.0 / tl.sqrt(var + eps)
        out = x * rsqrt * w
        out_ptrs = out_ptr + row_idx * stride_out_row + cols
        tl.store(out_ptrs, out, mask=mask)

    @triton.jit
    def _selective_scan_fwd_kernel(
        u_ptr,
        delta_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        out_ptr,
        B,
        L,
        D,
        N: tl.constexpr,
        stride_ub,
        stride_ul,
        stride_ud,
        stride_deltab,
        stride_deltal,
        stride_deltad,
        stride_ad,
        stride_an,
        stride_bb,
        stride_bl,
        stride_bn,
        stride_cb,
        stride_cl,
        stride_cn,
        stride_outb,
        stride_outl,
        stride_outd,
    ):
        pid = tl.program_id(0)
        b_idx = pid // D
        d_idx = pid % D
        state_cols = tl.arange(0, N)
        a_ptrs = a_ptr + d_idx * stride_ad + state_cols * stride_an
        a_val = tl.load(a_ptrs)
        h = tl.zeros((N,), dtype=tl.float32)
        for t in range(L):
            u_curr = tl.load(u_ptr + b_idx * stride_ub + t * stride_ul + d_idx * stride_ud)
            dt_curr = tl.load(delta_ptr + b_idx * stride_deltab + t * stride_deltal + d_idx * stride_deltad)
            b_ptrs = b_ptr + b_idx * stride_bb + t * stride_bl + state_cols * stride_bn
            b_curr = tl.load(b_ptrs)
            c_ptrs = c_ptr + b_idx * stride_cb + t * stride_cl + state_cols * stride_cn
            c_curr = tl.load(c_ptrs)
            da = tl.exp(dt_curr * a_val)
            dbu = dt_curr * b_curr * u_curr
            h = da * h + dbu
            y_curr = tl.sum(h * c_curr, axis=0)
            tl.store(out_ptr + b_idx * stride_outb + t * stride_outl + d_idx * stride_outd, y_curr)

def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if not HAS_TRITON or not x.is_cuda:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * weight
    x_flat = x.view(-1, x.shape[-1]).contiguous()
    weight_c = weight.contiguous()
    M, N = x_flat.shape
    out = torch.empty_like(x_flat)
    BLOCK_SIZE = triton.next_power_of_2(N)
    _rmsnorm_fwd_kernel[(M,)](
        x_flat,
        weight_c,
        out,
        x_flat.stride(0),
        out.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out.view_as(x)

def triton_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    a_mat: torch.Tensor,
    b_proj: torch.Tensor,
    c_proj: torch.Tensor,
) -> torch.Tensor:
    b, l, d = u.shape
    d_state = b_proj.shape[-1]
    if not HAS_TRITON or not u.is_cuda or d_state != 16:
        a_m = a_mat.view(1, d, d_state)
        b_bar = b_proj.float().unsqueeze(2)
        u_f = u.float().unsqueeze(-1)
        c_proj_f = c_proj.float()
        h = torch.zeros(b, d, d_state, device=u.device, dtype=torch.float32)
        ys = []
        d_squeeze = delta.squeeze(-1) if delta.dim() == 4 else delta
        for t in range(l):
            d_t = d_squeeze[:, t].unsqueeze(-1)
            da_t = torch.exp(d_t * a_m)
            dbu_t = (d_t * b_bar[:, t]) * u_f[:, t]
            h = da_t * h + dbu_t
            y_t = torch.einsum("bdn,bn->bd", h, c_proj_f[:, t])
            ys.append(y_t)
        return torch.stack(ys, dim=1).to(u.dtype)

    u_c = u.contiguous().float()
    delta_c = (delta.squeeze(-1) if delta.dim() == 4 else delta).contiguous().float()
    a_c = a_mat.view(d, d_state).contiguous().float()
    b_c = b_proj.contiguous().float()
    c_c = c_proj.contiguous().float()
    out = torch.empty(b, l, d, device=u.device, dtype=torch.float32)

    grid = (b * d,)
    _selective_scan_fwd_kernel[grid](
        u_c,
        delta_c,
        a_c,
        b_c,
        c_c,
        out,
        b,
        l,
        d,
        16,
        u_c.stride(0),
        u_c.stride(1),
        u_c.stride(2),
        delta_c.stride(0),
        delta_c.stride(1),
        delta_c.stride(2),
        a_c.stride(0),
        a_c.stride(1),
        b_c.stride(0),
        b_c.stride(1),
        b_c.stride(2),
        c_c.stride(0),
        c_c.stride(1),
        c_c.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out.to(u.dtype)
