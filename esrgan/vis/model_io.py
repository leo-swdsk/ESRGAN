import torch
from ..rrdb_ct_model import RRDBNet_CT


def load_model(model_path: str, scale: int, device: torch.device) -> torch.nn.Module:
    model = RRDBNet_CT(scale=scale).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model' in state and all(k in state for k in ['epoch', 'model']):
        state = state['model']
    model.load_state_dict(state)
    model.eval()
    return model


def infer_sr_slice(model: torch.nn.Module, lr_slice_bchw: torch.Tensor, device: torch.device, amp: bool = True) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if device.type == 'cuda' and amp:
            with torch.amp.autocast('cuda'):
                y = model(lr_slice_bchw.to(device))
        else:
            y = model(lr_slice_bchw.to(device))
    return y


