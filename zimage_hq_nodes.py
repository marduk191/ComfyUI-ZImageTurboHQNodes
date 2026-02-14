import random
import gc
import torch

import nodes
import comfy.utils
import comfy.samplers
import comfy.model_management as mm
from comfy.samplers import SchedulerHandler
from comfy.k_diffusion import sampling as k_diffusion_sampling


def _encode_conditioning(clip, text):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def _zero_conditioning(conditioning):
    zeroed = []
    for cond, meta in conditioning:
        out_meta = dict(meta)
        pooled = out_meta.get("pooled_output")
        if pooled is not None:
            out_meta["pooled_output"] = torch.zeros_like(pooled)
        zeroed.append([torch.zeros_like(cond), out_meta])
    return zeroed


def _safe_device(device_name):
    if device_name == "auto":
        return mm.get_torch_device()
    return torch.device(device_name)


def _compute_dtype(low_vram, device):
    return torch.float16 if low_vram and str(device).startswith("cuda") else torch.float32


def _capitan_basic(conditioning, strength, normalize, add_self_attention, mlp_hidden_mult, seed, low_vram, device):
    if not conditioning:
        return conditioning

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out = []
    dtype = _compute_dtype(low_vram, device)
    safe_mult = max(1, min(mlp_hidden_mult, 12))

    for cond, meta in conditioning:
        emb = cond.to(device, dtype=dtype)
        original_dtype = cond.dtype

        if normalize:
            mean = emb.mean(dim=-1, keepdim=True)
            std = emb.std(dim=-1, keepdim=True) + 1e-6
            emb = (emb - mean) / std

        dim = emb.shape[-1]
        hidden = dim * safe_mult
        mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden, device=device, dtype=dtype),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, dim, device=device, dtype=dtype),
        )
        torch.nn.init.kaiming_uniform_(mlp[0].weight, nonlinearity="relu")
        torch.nn.init.zeros_(mlp[0].bias)
        torch.nn.init.eye_(mlp[2].weight)
        torch.nn.init.zeros_(mlp[2].bias)

        refined = mlp(emb)
        mixed = emb + strength * (refined - emb)

        if add_self_attention and not low_vram:
            attn = torch.nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=8,
                batch_first=True,
                device=device,
                dtype=dtype,
            )
            attn_out, _ = attn(mixed, mixed, mixed)
            mixed = mixed + 0.3 * attn_out
            del attn, attn_out

        out.append((mixed.to("cpu", dtype=original_dtype), dict(meta)))

        del mlp, refined, mixed, emb
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    return out


def _capitan_advanced(conditioning, strength, detail_boost, preserve_original, attention_strength, high_pass_filter,
                      normalize, add_self_attention, mlp_hidden_mult, seed, low_vram, device):
    if not conditioning:
        return conditioning

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    out = []
    dtype = _compute_dtype(low_vram, device)
    safe_mult = max(1, min(mlp_hidden_mult, 16))

    for cond, meta in conditioning:
        emb = cond.to(device, dtype=dtype)
        original_dtype = cond.dtype

        if normalize:
            mean = emb.mean(dim=-1, keepdim=True)
            std = emb.std(dim=-1, keepdim=True) + 1e-6
            emb = (emb - mean) / std

        dim = emb.shape[-1]
        hidden = dim * safe_mult
        mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden, device=device, dtype=dtype),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, dim, device=device, dtype=dtype),
        )
        torch.nn.init.kaiming_uniform_(mlp[0].weight, nonlinearity="relu")
        torch.nn.init.zeros_(mlp[0].bias)
        torch.nn.init.eye_(mlp[2].weight)
        torch.nn.init.zeros_(mlp[2].bias)

        refined = mlp(emb)

        if detail_boost > 1.0:
            details = refined - emb
            details = torch.tanh(details * (detail_boost - 1.0))
            refined = emb + details

        if high_pass_filter:
            low_pass = torch.nn.functional.avg_pool1d(
                refined.transpose(1, 2), kernel_size=3, stride=1, padding=1
            ).transpose(1, 2)
            refined = refined + 0.4 * (refined - low_pass)
            del low_pass

        residual_scale = 1.0 / (1.0 + safe_mult * 0.05)
        mixed = emb + (strength * residual_scale) * (refined - emb)
        mixed = mixed * (1.0 - preserve_original) + emb * preserve_original

        if add_self_attention and not low_vram:
            attn = torch.nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=8,
                batch_first=True,
                device=device,
                dtype=dtype,
            )
            attn_out, _ = attn(mixed, mixed, mixed)
            mixed = mixed + attention_strength * attn_out
            del attn, attn_out

        out.append((mixed.to("cpu", dtype=original_dtype), dict(meta)))

        del mlp, refined, mixed, emb
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    return out


def _resolve_sampler_scheduler(sampling_profile):
    samplers = set(comfy.samplers.KSampler.SAMPLERS)
    schedulers = set(comfy.samplers.SCHEDULER_NAMES)

    if sampling_profile == "capitan_flow":
        sampler = "euler_flow" if "euler_flow" in samplers else "euler"
        if "capitanZiT" in schedulers:
            scheduler = "capitanZiT"
        elif "zimage_turbo" in schedulers:
            scheduler = "zimage_turbo"
        else:
            scheduler = "simple"
    elif sampling_profile == "zflow_linear":
        sampler = "euler_flow" if "euler_flow" in samplers else "euler"
        if "zimage_turbo" in schedulers:
            scheduler = "zimage_turbo"
        elif "capitanZiT" in schedulers:
            scheduler = "capitanZiT"
        else:
            scheduler = "simple"
    else:
        sampler = "res_multistep"
        scheduler = "simple"

    return sampler, scheduler


def _sample_euler_flow(model, x, sigmas, extra_args=None, callback=None, disable=None):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})

        if sigma_next == 0:
            x = denoised
            break

        if sigma > 1e-6:
            ratio = sigma_next / sigma
            x = ratio * x + (1.0 - ratio) * denoised
        else:
            x = denoised

    return x


def _register_local_zimage_turbo_sampling():
    # Self-contained sampler integration: no external node packs required.
    if "euler_flow" not in comfy.samplers.KSampler.SAMPLERS:
        comfy.samplers.KSampler.SAMPLERS.append("euler_flow")
    setattr(k_diffusion_sampling, "sample_euler_flow", _sample_euler_flow)

    if "capitanZiT" not in comfy.samplers.SCHEDULER_NAMES:
        comfy.samplers.SCHEDULER_NAMES.append("capitanZiT")
    if "zimage_turbo" not in comfy.samplers.SCHEDULER_NAMES:
        comfy.samplers.SCHEDULER_NAMES.append("zimage_turbo")

    def _linear_sigmas(_model, steps):
        device = mm.get_torch_device()
        return torch.linspace(1.0, 0.0, steps + 1).to(device)

    comfy.samplers.SCHEDULER_HANDLERS["capitanZiT"] = SchedulerHandler(_linear_sigmas, use_ms=True)
    comfy.samplers.SCHEDULER_HANDLERS["zimage_turbo"] = SchedulerHandler(_linear_sigmas, use_ms=True)


_register_local_zimage_turbo_sampling()


class ZImageTurboConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        devices = ["auto", "cpu"]
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")

        return {
            "required": {
                "clip": ("CLIP",),
                "subject": ("STRING", {"multiline": True, "default": "portrait photo of a person"}),
                "style": ("STRING", {"multiline": True, "default": "cinematic, natural skin texture"}),
                "lighting": ("STRING", {"multiline": True, "default": "soft directional light"}),
                "details": ("STRING", {"multiline": True, "default": "ultra detailed, clean composition"}),
                "enhancement_profile": (["none", "capitan_daily", "capitan_literal"], {"default": "capitan_daily"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "device": (devices, {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "positive_prompt")
    FUNCTION = "build"
    CATEGORY = "zimage_turbo/hq"

    def build(self, clip, subject, style, lighting, details, enhancement_profile, seed, low_vram, device):
        positive_prompt = f"{subject}, {style}, {lighting}, {details}, masterpiece, high quality"
        positive = _encode_conditioning(clip, positive_prompt)

        if enhancement_profile != "none":
            dev = _safe_device(device)
            # Repo-derived stack: basic glue + advanced literal detail, with VRAM-safe multipliers.
            positive = _capitan_basic(
                positive,
                strength=0.08 if enhancement_profile == "capitan_daily" else 0.06,
                normalize=True,
                add_self_attention=True,
                mlp_hidden_mult=3 if enhancement_profile == "capitan_daily" else 4,
                seed=seed,
                low_vram=low_vram,
                device=dev,
            )
            positive = _capitan_advanced(
                positive,
                strength=0.04 if enhancement_profile == "capitan_daily" else 0.06,
                detail_boost=1.9 if enhancement_profile == "capitan_daily" else 2.2,
                preserve_original=0.40 if enhancement_profile == "capitan_daily" else 0.45,
                attention_strength=0.05,
                high_pass_filter=True,
                normalize=True,
                add_self_attention=False,
                mlp_hidden_mult=8 if enhancement_profile == "capitan_daily" else 12,
                seed=seed,
                low_vram=low_vram,
                device=dev,
            )

        negative = _zero_conditioning(_encode_conditioning(clip, ""))
        return (positive, negative, positive_prompt)


class ZImageTurboLatentInit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution": (["1024", "1280", "1536"], {"default": "1024"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height")
    FUNCTION = "build"
    CATEGORY = "zimage_turbo/hq"

    def build(self, resolution, batch_size):
        size = int(resolution)
        latent = torch.zeros([batch_size, 16, size // 8, size // 8], device="cpu")
        return ({"samples": latent}, size, size)


class ZImageTurboSeedControl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_mode": (["random", "fixed"], {"default": "random"}),
                "fixed_seed": ("INT", {"default": 13371337, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "refine_seed_offset": ("INT", {"default": 101, "min": 0, "max": 1000000}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("base_seed", "refine_seed")
    FUNCTION = "seeds"
    CATEGORY = "zimage_turbo/hq"

    def seeds(self, seed_mode, fixed_seed, refine_seed_offset):
        base = fixed_seed if seed_mode == "fixed" else random.randint(0, 0xFFFFFFFF)
        return (base, base + refine_seed_offset)


class ZImageTurboSamplingPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["base_ultra", "base_balanced", "refine_subtle", "refine_normal", "refine_strong"], {"default": "base_ultra"}),
                "sampling_profile": (["tongyi_default", "capitan_flow", "zflow_linear"], {"default": "capitan_flow"}),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("steps", "cfg", "denoise", "sampler_name", "scheduler_name")
    FUNCTION = "plan"
    CATEGORY = "zimage_turbo/hq"

    def plan(self, mode, sampling_profile):
        if mode == "base_ultra":
            steps, cfg, denoise = 9, 1.0, 1.0
        elif mode == "base_balanced":
            steps, cfg, denoise = 8, 1.0, 1.0
        elif mode == "refine_subtle":
            steps, cfg, denoise = 8, 1.0, 0.25
        elif mode == "refine_normal":
            steps, cfg, denoise = 9, 1.0, 0.32
        else:
            steps, cfg, denoise = 10, 1.0, 0.40

        sampler_name, scheduler_name = _resolve_sampler_scheduler(sampling_profile)
        return (steps, cfg, denoise, sampler_name, scheduler_name)


class ZImageTurboSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 13371337, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "mode": (["base_ultra", "base_balanced", "refine_subtle", "refine_normal", "refine_strong"], {"default": "base_ultra"}),
                "sampling_profile": (["tongyi_default", "capitan_flow", "zflow_linear"], {"default": "capitan_flow"}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("samples", "steps", "cfg", "denoise", "sampler_name", "scheduler_name")
    FUNCTION = "sample"
    CATEGORY = "zimage_turbo/hq"

    def sample(self, model, positive, negative, latent_image, seed, mode, sampling_profile):
        steps, cfg, denoise, sampler_name, scheduler_name = ZImageTurboSamplingPlan().plan(mode, sampling_profile)

        out = nodes.common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler_name,
            positive,
            negative,
            latent_image,
            denoise=denoise,
        )
        return (out[0], steps, cfg, denoise, sampler_name, scheduler_name)


class ZImageTurboTwoPassRefiner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 13371438, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05}),
                "strength": (["subtle", "normal", "strong"], {"default": "normal"}),
                "sampling_profile": (["tongyi_default", "capitan_flow", "zflow_linear"], {"default": "capitan_flow"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "refine"
    CATEGORY = "zimage_turbo/hq"

    def refine(self, model, positive, negative, vae, image, seed, upscale_by, strength, sampling_profile):
        mode = {
            "subtle": "refine_subtle",
            "normal": "refine_normal",
            "strong": "refine_strong",
        }[strength]
        steps, _, denoise, sampler_name, scheduler_name = ZImageTurboSamplingPlan().plan(mode, sampling_profile)

        pixels = image.movedim(-1, 1)
        upscaled = comfy.utils.common_upscale(
            pixels,
            max(8, int(pixels.shape[3] * upscale_by)),
            max(8, int(pixels.shape[2] * upscale_by)),
            "lanczos",
            "disabled",
        ).movedim(1, -1)

        latent = vae.encode(upscaled[:, :, :, :3])
        sampled = nodes.common_ksampler(
            model,
            seed,
            steps,
            1.0,
            sampler_name,
            scheduler_name,
            positive,
            negative,
            {"samples": latent},
            denoise=denoise,
        )[0]
        decoded = vae.decode(sampled["samples"])
        return (decoded,)


NODE_CLASS_MAPPINGS = {
    "ZImageTurboConditioning": ZImageTurboConditioning,
    "ZImageTurboLatentInit": ZImageTurboLatentInit,
    "ZImageTurboSeedControl": ZImageTurboSeedControl,
    "ZImageTurboSamplingPlan": ZImageTurboSamplingPlan,
    "ZImageTurboSampler": ZImageTurboSampler,
    "ZImageTurboTwoPassRefiner": ZImageTurboTwoPassRefiner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZImageTurboConditioning": "ZImage Turbo Conditioning",
    "ZImageTurboLatentInit": "ZImage Turbo Latent Init",
    "ZImageTurboSeedControl": "ZImage Turbo Seed Control",
    "ZImageTurboSamplingPlan": "ZImage Turbo Sampling Plan",
    "ZImageTurboSampler": "ZImage Turbo Sampler",
    "ZImageTurboTwoPassRefiner": "ZImage Turbo Two Pass Refiner",
}
