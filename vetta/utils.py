from __future__ import annotations

import glob
import json
import os
import shutil

import numpy as np
import numpy.typing as npt
import PIL.Image

try:
    import torch
except ModuleNotFoundError:
    torch = None


def require_optional_dependency(
    dependency,
    package_name: str,
    purpose: str,
) -> None:
    if dependency is None:
        raise ImportError(
            f"Optional dependency '{package_name}' is required for {purpose}."
        )


def to_array(
    x: npt.NDArray | "torch.Tensor" | dict | list | int | float,
) -> npt.NDArray | dict | list:
    if torch is not None and (isinstance(x, torch.Tensor) or torch.is_tensor(x)):
        return x.detach().cpu().numpy()
    if isinstance(x, list):
        return [to_array(item) for item in x]
    if isinstance(x, dict):
        return {key: to_array(value) for key, value in x.items()}
    if type(x) in [int, float]:
        return np.array([x])
    return x


def to_tensor(
    x: npt.NDArray | "torch.Tensor" | list | dict,
    to_cuda: bool = False,
) -> "torch.Tensor" | dict | list | None:
    if torch is None:
        raise ImportError("Optional dependency 'torch' is required for to_tensor().")

    if isinstance(x, np.ndarray):
        if x.dtype == np.object_ and np.any(x == None):  # noqa: E711
            return None
        x = torch.tensor(x)
        if to_cuda:
            x = x.cuda()
        return x
    if isinstance(x, list):
        return [to_tensor(item, to_cuda=to_cuda) for item in x]
    if isinstance(x, dict):
        return {key: to_tensor(value, to_cuda=to_cuda) for key, value in x.items()}
    if type(x) in [int, float, np.float64, np.float32, np.int64, np.int32]:
        return torch.tensor([x])
    if isinstance(x, torch.Tensor):
        if to_cuda:
            x = x.cuda()
        return x
    if x is None:
        return None
    raise Exception("unhandled type " + str(type(x)))


def make_json_serializable(json_data):
    def convert(obj):
        if isinstance(obj, float):
            return obj
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if torch is not None and isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        if torch is not None and isinstance(obj, torch.nn.Module):
            return obj.__repr__()
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, (set, list, tuple)):
            return type(obj)([convert(item) for item in obj])
        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        return str(obj)

    return convert(json_data)


def check_dtype(arr, key_str, dtypes_np, dtypes_pt, fail_on_cast=False):
    if arr is None:
        return None
    if type(arr) in [int, float, np.int32, np.int64, np.float32, np.float64]:
        if fail_on_cast:
            raise Exception("array " + str(key_str) + " is of type " + str(type(arr)))
        arr = to_array(arr)
    elif (
        torch is not None
        and type(arr) in [torch.int32, torch.int64, torch.float32, torch.float64]
    ):
        if fail_on_cast:
            raise Exception("array " + str(key_str) + " is of type " + str(type(arr)))
        arr = to_tensor(arr)

    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        if arr.dtype != dtypes_np[key_str]:
            if fail_on_cast:
                raise Exception(
                    "array " + str(key_str) + " is of dtype " + str(arr.dtype)
                )
            arr = to_array(arr)
            target_type = dtypes_np[key_str]
            arr = arr.astype(target_type)
    elif torch is not None and isinstance(arr, torch.Tensor):
        if arr.dtype != dtypes_pt[key_str]:
            if fail_on_cast:
                raise Exception(
                    "array " + str(key_str) + " is of dtype " + str(arr.dtype)
                )
            tgt_type = dtypes_pt[key_str]
            if tgt_type == torch.int32:
                arr = to_tensor(arr).int()
            elif tgt_type == torch.float32:
                arr = to_tensor(arr).float()
            else:
                raise Exception("type not supported " + str(tgt_type))

    return arr


class temporary_seed:
    def __init__(self, seed):
        self.seed = seed
        self.backup = None

    def __enter__(self):
        self.backup = np.random.randint(2**32 - 1, dtype=np.uint32)
        np.random.seed(self.seed)

    def __exit__(self, *_):
        np.random.seed(self.backup)


def listdir_fullpath(directory: str, do_sort: bool = True) -> list[str] | None:
    if not os.path.exists(directory):
        return None
    paths = [os.path.join(directory, element) for element in os.listdir(directory)]
    if do_sort:
        paths = sorted(paths)
    return paths


def mkdir_custom(
    directory,
    dir_exists_mode="delete",
    parent_does_not_exist_mode="make_parent",
):
    accepted_dir_exists_modes = ["delete", "delete_contents", "fail", "return"]
    accepted_parent_does_not_exist_modes = ["make_parent", "fail"]

    if dir_exists_mode not in accepted_dir_exists_modes:
        raise Exception(
            "dir_exists_mode "
            + str(dir_exists_mode)
            + " not in "
            + str(accepted_dir_exists_modes)
        )

    if parent_does_not_exist_mode not in accepted_parent_does_not_exist_modes:
        raise Exception(
            "parent_does_not_exist_mode "
            + str(parent_does_not_exist_mode)
            + " not in "
            + str(accepted_parent_does_not_exist_modes)
        )

    parent_dir = os.path.abspath(os.path.join(directory, os.pardir))
    if not os.path.exists(parent_dir):
        if parent_does_not_exist_mode == "make_parent":
            os.makedirs(parent_dir)
        elif parent_does_not_exist_mode == "fail":
            raise Exception("parent directory " + str(parent_dir) + " does not exist")

    if os.path.exists(directory):
        if dir_exists_mode == "delete":
            shutil.rmtree(directory)
        elif dir_exists_mode == "delete_contents":
            for subelement in listdir_fullpath(directory):
                shutil.rmtree(subelement)
        elif dir_exists_mode == "fail":
            raise Exception("directory " + str(directory) + " exists")
        elif dir_exists_mode == "return":
            return directory

    os.makedirs(directory)
    return directory


def mkdir_force(directory, force=False):
    dir_exists_mode = "delete" if force else "fail"
    return mkdir_custom(directory, dir_exists_mode=dir_exists_mode)


def remove_custom(path, verbose=False, fail_verbose=True):
    if not os.path.exists(path):
        if fail_verbose:
            print("path does not exist: " + str(path), flush=True)
        return False
    if os.path.isdir(path):
        if verbose:
            print("deleting directory: " + str(path), flush=True)
        shutil.rmtree(path)
        return True
    if verbose:
        print("deleting file : " + str(path), flush=True)
    os.remove(path)
    return True


def find_files(directory, pattern):
    full_pattern = os.path.join(directory, pattern)
    return glob.glob(full_pattern)


def find_files_0(directory, pattern):
    matching_files = find_files(directory, pattern)
    if len(matching_files) >= 1:
        return matching_files[0]
    return None


def str_to_bool(val: str) -> bool:
    value = val.lower().strip()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"unknown boolean string {val!r}")


def randn_like(x: "torch.Tensor" | npt.NDArray | list) -> "torch.Tensor" | npt.NDArray:
    if torch is not None and torch.is_tensor(x):
        return torch.randn_like(x)
    if isinstance(x, np.ndarray):
        return np.random.randn(*x.shape).astype(x.dtype)
    if isinstance(x, list):
        return np.random.randn(*x)
    raise Exception("unhandled type " + str(type(x)))


def take_along_axis(x, index, axis):
    if isinstance(x, np.ndarray):
        return np.take(x, index, axis=axis)
    if torch is not None and isinstance(x, torch.Tensor):
        if isinstance(index, int):
            index = torch.tensor([index]).to(x.device)
        return torch.index_select(x, dim=axis, index=index).squeeze(axis)
    raise Exception("unhandled type " + str(type(x)))


def printf(string: str | None = None) -> None:
    if string is None:
        print("", flush=True)
    else:
        print(string, flush=True)


def unstack(a, axis=0):
    return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis=axis)]


def liftsin(x, oct):
    if isinstance(x, np.ndarray):
        return np.sin(2 * np.pi * oct * x).astype(np.float32)
    if torch is not None and isinstance(x, torch.Tensor):
        return torch.sin(2 * np.pi * oct * x).float()
    raise Exception("unhandled type " + str(type(x)))


def liftcos(x, oct):
    if isinstance(x, np.ndarray):
        return np.cos(2 * np.pi * oct * x).astype(np.float32)
    if torch is not None and isinstance(x, torch.Tensor):
        return torch.cos(2 * np.pi * oct * x).float()
    raise Exception("unhandled type " + str(type(x)))


def lift(x, oct, lifter):
    if lifter == "sin":
        return liftsin(x, oct)
    if lifter == "cos":
        return liftcos(x, oct)
    raise Exception("unhandled lifter " + str(lifter))


def add_octaves(
    x,
    octaves,
    dim,
    channels=None,
    ret_type=None,
    lifters=["sin", "cos"],
    mask=None,
    include_base=True,
):
    del ret_type, mask
    if channels is None:
        channels = list(range(x.shape[dim]))
    moved = np.moveaxis(x, dim, -1) if isinstance(x, np.ndarray) else torch.movedim(x, dim, -1)
    parts = []
    if include_base:
        parts.append(moved)
    for channel in channels:
        channel_vals = moved[..., channel : channel + 1]
        for octv in octaves:
            for lifter_name in lifters:
                parts.append(lift(channel_vals, octv, lifter_name))
    out = np.concatenate(parts, axis=-1) if isinstance(moved, np.ndarray) else torch.cat(parts, dim=-1)
    if isinstance(out, np.ndarray):
        return np.moveaxis(out, -1, dim)
    return torch.movedim(out, -1, dim)


def powers_of_two(max_oct):
    if (max_oct & (max_oct - 1)) != 0 or max_oct <= 0:
        raise ValueError("max_oct should be a positive power of 2")
    powers = []
    power = 0
    while 2**power <= max_oct:
        powers.append(2**power)
        power += 1
    return powers


def save_array_txt(path, array_np, dtype=None, width=8):
    array_np = to_array(array_np)
    array_np = np.atleast_1d(array_np)
    if dtype is None:
        dtype = array_np.dtype
    if dtype in [np.float32, np.float64]:
        fmt = "% " + str(width) + ".6f"
    elif dtype in [np.int32, np.int64]:
        fmt = "%-" + str(width) + "d"
    else:
        raise Exception("dtype " + str(dtype) + " not supported")
    np.savetxt(path, array_np, fmt=fmt)


def save_string(path, string):
    with open(path, "w") as f:
        f.write(string)


def save_array(path, array_np, compress=False):
    if compress:
        np.savez_compressed(path, array_np)
    else:
        np.save(path, array_np)


def load_array(path):
    if path.split(".")[-1] == "npz":
        data = np.load(path, allow_pickle=True)
        key_str = list(data.keys())[0]
        return data[key_str]
    return np.load(path, allow_pickle=True)


def save_image(path, array_np):
    if len(array_np.shape) not in [2, 3]:
        raise Exception("unhandled shape " + str(array_np.shape))
    if len(array_np.shape) == 2:
        array_np = np.expand_dims(array_np, axis=-1)
    if len(array_np.shape) == 3 and array_np.shape[2] == 1:
        array_np = np.concatenate([array_np] * 3, axis=-1)
    PIL.Image.fromarray(array_np).save(path)


def load_image(path):
    return np.asarray(PIL.Image.open(path))


def save_json(path, json_data, indent=4):
    json_data = make_json_serializable(json_data)
    json_string = json.dumps(json_data, indent=indent)
    with open(path, "w") as f:
        f.write(json_string)


def load_json(path):
    with open(path, "r") as f:
        return json.loads(f.read())


__all__ = [
    "add_octaves",
    "check_dtype",
    "find_files",
    "find_files_0",
    "lift",
    "liftcos",
    "liftsin",
    "listdir_fullpath",
    "load_array",
    "load_image",
    "load_json",
    "make_json_serializable",
    "mkdir_custom",
    "mkdir_force",
    "np",
    "npt",
    "powers_of_two",
    "printf",
    "randn_like",
    "remove_custom",
    "require_optional_dependency",
    "save_array",
    "save_array_txt",
    "save_image",
    "save_json",
    "save_string",
    "str_to_bool",
    "take_along_axis",
    "temporary_seed",
    "to_array",
    "to_tensor",
    "torch",
    "unstack",
]
