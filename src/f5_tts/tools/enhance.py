from resemble_enhance.enhancer.inference import enhance
import torch

def enhance_audio(wav: torch.Tensor, sr: int):
    try:
        wav = wav.mean(dim=0)
        wav = wav.to(torch.float32)
        wav, new_sr = enhance(wav, sr, device="cuda")
        wav = wav.cpu()
        torch.cuda.empty_cache()
        return wav, new_sr
    except Exception as e:
        torch.cuda.empty_cache()
        raise e