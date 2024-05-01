from PIL import Image
from IPython.display import display
import torch as th

from download import load_checkpoint
from create_model import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)


class DiffusionModel:
    def __init__(self):
        self._cuda = th.cuda.is_available()
        self._device = th.device('cuda' if self._cuda else 'cpu')

        self.model_options = model_and_diffusion_defaults()
        self.upsampler_options = model_and_diffusion_defaults_upsampler()

        # Define static options
        self.model_options['use_fp16'] = self._cuda
        self.model_options['timestep_respacing'] = '100'

        self.upsampler_options['use_fp16'] = self._cuda
        self.upsampler_options['timestep_respacing'] = 'fast27'
        self.batch_size = 1
        self.guidance_scale = 3.0

        self.model = None
        self.diffusion = None

        self.up_model = None
        self.up_difusion = None

    def set_model_variables(self, **kwargs):
        for key, value in kwargs.items():
            self.model_options[key] = value

    def create_models(self):
        # Diffusion definition
        self.model, self.diffusion = create_model_and_diffusion(**self.model_options)
        self.model.eval()

        if self._cuda:
            self.model.convert_to_fp16()

        self.model.to(self._device)
        self.model.load_state_dict(load_checkpoint('base', device=self._device))

        # Upsampler definition
        up_model, up_difusion = create_model_and_diffusion(**self.upsampler_options)
        up_model.eval()
        if self._cuda:
            up_model.convert_to_fp16()
        self.up_model.to(self._device)
        self.up_model.load_state_dict(load_checkpoint('upsample', device=self._device))

    def model_fn(self, x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = self.model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)

    def create_image(self, prompt: str) -> Image.Image:
        # static static variable, dont change please :)
        upsample_temp = 0.997
        tokens = self.model.tokenizer.encode(prompt)
        tokens, mask = self.model.tokenizer.padded_tokens_and_mask(
            tokens, self.model_options['text_ctx']
        )
        full_batch_size = self.batch_size * 2
        uncond_tokens, uncond_mask = self.model.tokenizer.padded_tokens_and_mask(
            [], self.model_options['text_ctx']
        )
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * self.batch_size + [uncond_tokens] * self.batch_size, device=self._device
            ),
            mask=th.tensor(
                [mask] * self.batch_size + [uncond_mask] * self.batch_size,
                dtype=th.bool,
                device=self._device,
            ),
        )
        self.model.del_cache()
        samples = self.diffusion.p_sample_loop(
            self.model_fn,
            (full_batch_size, 3, self.model_options["image_size"], self.model_options["image_size"]),
            device=self._device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:self.batch_size]
        self.model.del_cache()

        scaled = ((samples + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
        reshaped = scaled.permute(2, 0, 3, 1).reshape([samples.shape[2], -1, 3])
        return Image.fromarray(reshaped.numpy())
