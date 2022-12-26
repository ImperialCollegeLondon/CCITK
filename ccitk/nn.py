__all__ = [
	"state_dict_checkpoint_to_pickle",
	"pickle_checkpoint_to_state_dict",
]

import torch
from pathlib import Path
from typing import Union


def state_dict_checkpoint_to_pickle(
		network: torch.nn.Module, checkpoint_path: Path, output_path: Path, device: Union[str, int]
):
	"""Convert state dict format to pickle format

	Args:
		network: the network that the state dict represents.
		checkpoint_path: the checkpoint path of the state dict
		output_path: output path
		device: device that state dict was stored as, gpu or cpu

	Returns:
		None
	"""
	checkpoint = torch.load(str(Path(checkpoint_path)), map_location=torch.device(device))
	network.load_state_dict(checkpoint)
	if isinstance(device, int):
		network.cuda(device)
	torch.save(network, str(output_path))


def pickle_checkpoint_to_state_dict(checkpoint_path: Path, output_path: Path, device: Union[str, int]):
	"""Convert pickle format of the models to state dict

	Args:
		checkpoint_path: path to pickle format checkpoint
		output_path: output path
		device: device to store state dict as, gpu or cpu

	Returns:
		None
	"""
	model = torch.load(str(checkpoint_path))
	if isinstance(device, int):
		model.cuda(device)
	torch.save(model.state_dict(), str(output_path))
