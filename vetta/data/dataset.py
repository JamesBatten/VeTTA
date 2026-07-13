from __future__ import annotations

import contextlib
import os
import random
import time
import timeit

from .common import _require_optional_dependency
from .element import Element
from .trees import Tree
from .dataset_config import copy_tree_dataset_config
from .dataset_config import normalise_tree_dataset_config
from .dataset_config import resolve_tree_dataset_seed
from .transforms import make_transforms
from . import dataset_ops
from .field_specs import ELEMENT_OPTIONAL_PAYLOAD_SPECS
from .field_specs import ELEMENT_SPLIT_SUPERVISION_SPECS
from .field_specs import field_names
from .field_specs import select_known_fields
import numpy as np
import numpy.typing as npt

from vetta.utils import listdir_fullpath
from vetta.utils import load_array
from vetta.utils import save_array
from vetta.utils import to_tensor

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import torch
    from torch.utils.data import IterableDataset
except ModuleNotFoundError:
    torch = None

    class IterableDataset:
        pass

try:
    import zmq
except ModuleNotFoundError:
    zmq = None

SPLIT_PAYLOAD_FIELD_NAMES = (
    "pivot_idx",
    "query_idx",
    "query",
    "query_children",
)
SPLIT_LHS_FIELD_NAMES = (
    "branch_mask_lhs",
    "branches_lhs",
    "depth_lhs",
    "n_children_lhs",
    "node_mask_lhs",
    "topology_lhs",
    "edges_lhs",
    "edges_mask_lhs",
    "radius_lhs",
)
SPLIT_TO_ELEMENT_SPECS = tuple(
    spec
    for spec in (*ELEMENT_OPTIONAL_PAYLOAD_SPECS, *ELEMENT_SPLIT_SUPERVISION_SPECS)
    if spec.name in (*SPLIT_PAYLOAD_FIELD_NAMES, *SPLIT_LHS_FIELD_NAMES)
)
SPLIT_TO_ELEMENT_FIELDS = field_names(SPLIT_TO_ELEMENT_SPECS)
REQUIRED_SPLIT_FIELDS = tuple(
    field for field in SPLIT_TO_ELEMENT_FIELDS if field != "radius_lhs"
)

# --- begin public data: dataset.py ---
class TreeDataset(IterableDataset):

    def __init__(self, config=None):
        _require_optional_dependency(torch, "torch")
        super(TreeDataset, self).__init__()
        self.config = normalise_tree_dataset_config(config)

        worker_id = None
        winfo = torch.utils.data.get_worker_info()
        if winfo is not None:
            worker_id = winfo.id

        seed = resolve_tree_dataset_seed(self.config, worker_id=worker_id)

        np.random.seed(seed)
        random.seed(seed)

        self.transform = make_transforms(self.config)

        self.zmq_started = False

        self.candidate_noise = None


    @classmethod
    def default_config(cls):
        return copy_tree_dataset_config()


    def start_zmq(self):
        _require_optional_dependency(zmq, "pyzmq")
        if self.config['verbose']:
            print("TreeDataset::start_zmq()", flush=True)
        self.context = zmq.Context()
        self.context.setsockopt(zmq.LINGER, 0)
        self.clients = []
        self.pollers = []
        for k in range(0, self.config['n_servers']):
            self.reset_client(k)
        if self.config['verbose']:
            print("TreeDataset.zmq_started=True", flush=True)
        self.zmq_started = True


    def __del__(self):
        if hasattr(self, 'clients'):
            for k in range(0, len(self.clients)):
                self.clients[k].close()
        if hasattr(self, 'context') and self.context is not None:
            self.context.destroy()


    def pull_item(self):
        tree = self._pull_tree_from_source()
        return {
            'tree': tree
        }


    def _pull_tree_from_source(self) -> Tree | None:
        if not self.zmq_started:
            self.start_zmq()

        client_idx = self._select_client_idx()
        return self._pull_tree_from_client(client_idx)


    def _select_client_idx(self) -> int:
        return int(np.random.randint(0, len(self.clients)))


    def _pull_tree_from_client(self, client_idx: int) -> Tree | None:
        client = self.clients[client_idx]
        poller = self.pollers[client_idx]

        polling = True
        tree = None
        start_time = timeit.default_timer()
        while polling:
            socks = dict(poller.poll(self.config['poll_time']))
            vget = socks.get(client)
            if vget == zmq.POLLIN:
                if self.config['receive_mode'] == 'tree':
                    tree = Tree.recv_zmq(
                        client, recv_segmentation=self.config['load_segmentation'],
                        recv_radius=self.config['load_radius']
                    )
                else:
                    raise ValueError("Unknown receive mode: {}".format(self.config['receive_mode']))
                self._request_next_item(client)
            else:
                time.sleep(self.config['poll_time'])
            if tree is not None:
                polling = False
            if timeit.default_timer() - start_time > self.config['tpn_timeout']:
                self.reset_client(client_idx)
                polling = False

        return tree


    def reset_client(self, client_idx: int) -> None:
        _require_optional_dependency(zmq, "pyzmq")
        endpoint = self._client_endpoint(client_idx)
        if self.config['verbose']:
            print("connecting client to", endpoint, flush=True)

        if client_idx < len(getattr(self, 'clients', [])):
            with contextlib.suppress(Exception):
                self.clients[client_idx].close()
        client = self.context.socket(zmq.REQ)
        client.connect(endpoint)
        poller = zmq.Poller()
        poller.register(client, zmq.POLLIN)
        self._request_next_item(client)

        if client_idx < len(self.clients):
            self.clients[client_idx] = client
            self.pollers[client_idx] = poller
        else:
            self.clients.append(client)
            self.pollers.append(poller)


    def _client_endpoint(self, client_idx: int) -> str:
        port = self.config['start_port'] + client_idx
        return "tcp://127.0.0.1:{}".format(port)


    def _request_next_item(self, client) -> None:
        client.send_string("requesting item")


    def tree_to_dict(self, tree) -> dict | None:
        element = self._element_from_tree(tree)
        element = self._apply_supervision_fields(element, tree)
        element = self._apply_pre_tensor_processing(element, tree)
        element = self._apply_tensor_transform(element)
        element = self._apply_candidate_fields(element)

        if element is None:
            if self.config['show_warnings']:
                print("warning: element is None", flush=True)
            return None

        tdict = element.tensor_dict() # pytorch collate requires tensor dict
        return tdict


    def _element_from_tree(self, tree: Tree) -> Element:
        element = tree.to_element()
        return TreeDataset.add_metadata(element, tree)


    def _apply_supervision_fields(self, element: Element, tree: Tree) -> Element:
        if self.config['add_splits']:
            element = TreeDataset.add_splits(element, tree, self.config)
        return element


    def _apply_pre_tensor_processing(self, element: Element, tree: Tree) -> Element:
        if self.config['normalise']:
            element = TreeDataset.normalise(element, self.config)

        if self.config['add_semi']:
            element = TreeDataset.add_semi(element, tree, self.config)
            # add the query edge index after the add_semi call
            element = TreeDataset.add_qeidx(element)

        if self.config['filter_non_proximal']:
            element = TreeDataset.filter_non_proximal(element, tree, self.config)

        return self._validate_and_augment_element(element)


    def _validate_and_augment_element(self, element: Element) -> Element:
        element.check_dtypes(fail_on_cast=False)
        element.check_arrays()

        if self.config['pad_before_aug']:
            element = TreeDataset.preaug_pad(element, self.config)

        if self.config['mode'] == 'train' and self.config['allow_augmentation']:
            element = TreeDataset.augment_pos(
                element, self.config
            )

        return element


    def _apply_tensor_transform(self, element: Element) -> Element | None:
        return self.transform(element)


    def _apply_candidate_fields(self, element: Element | None) -> Element | None:
        if self.config['add_candidates']:
            raise NotImplementedError(
                "candidate generation has been removed from vetta.data"
            )
        return element


    def try_put_next(self) -> dict | None:
        item = self.pull_item()
        if item is None:
            return None
        if item['tree'] is None:
            return None
        return self.tree_to_dict(item['tree'])


    @classmethod
    def add_metadata(cls, element: Element, tree: Tree) -> Element:
        if tree.metadata is not None:
            metadata_dict = TreeDataset.make_metadata(tree)
            element.update_fields(metadata_dict)
        return element


    @classmethod
    def add_splits(cls, element: Element, tree: Tree, config: dict) -> Element:
        split_dict = tree.random_decoding_split(
            minpos=config['min_pivot_pos'],
            maxpos=config['max_pivot_pos'],
            order_seed=config['order_seed']
        )
        for field in REQUIRED_SPLIT_FIELDS:
            split_dict[field]
        split_fields = select_known_fields(split_dict, SPLIT_TO_ELEMENT_SPECS)
        split_fields.setdefault("radius_lhs", None)
        element.update_fields(split_fields)
        element.global_pos_lhs = element.global_pos * element.node_mask_lhs.reshape(-1, 1)

        return element


    @classmethod
    def normalise(cls, element: Element, config: dict) -> Element:
        has_pos_lhs = element.global_pos_lhs is not None
        em, center = cls.compute_norm_params(
            pos=element.global_pos, node_mask=element.node_mask
        )
        pos, radius = cls.normalise_aux(
            element.global_pos, element.radius, element.node_mask, em,
            center, config['norm_domain']
        )
        element.global_pos = pos
        if radius is not None:
            element.radius = radius
        if has_pos_lhs:
            pos_lhs, radius_lhs = cls.normalise_aux(
                element.global_pos_lhs, element.radius_lhs,
                element.node_mask_lhs, em, center, config['norm_domain']
            )
            element.global_pos_lhs = pos_lhs
            if radius_lhs is not None:
                element.radius_lhs = radius_lhs
        return element


    @classmethod
    def compute_norm_params(cls,
                            pos: npt.NDArray[np.float32],
                            node_mask: npt.NDArray[np.float32]
                            ) -> tuple[float, npt.NDArray[np.float32]]:
        params = dataset_ops.compute_norm_params(pos, node_mask)
        return params.extent, params.center


    @classmethod
    def normalise_aux(cls,
                      pos: npt.NDArray[np.float32],
                      radius: npt.NDArray[np.float32] | None,
                      node_mask: npt.NDArray[np.float32],
                      em: float,
                      center: npt.NDArray[np.float32],
                      norm_domain: tuple
                      ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:
        return dataset_ops.normalise_aux(pos, radius, node_mask, em, center, norm_domain)


    @classmethod
    def denormalise_aux(cls,
                        pos: npt.NDArray[np.float32],
                        node_mask: npt.NDArray[np.float32],
                        em: float,
                        center: npt.NDArray[np.float32],
                        norm_domain: tuple,
                        radius: npt.NDArray[np.float32] | None
                        ) -> npt.NDArray[np.float32]:
        return dataset_ops.denormalise_aux(pos, node_mask, em, center, norm_domain, radius)


    @classmethod
    def remove_batch_dim(cls, full_tree_dict: dict):
        return dataset_ops.remove_batch_dim(full_tree_dict)


    @classmethod
    def add_batch_dim(cls, full_tree_dict: dict):
        return dataset_ops.add_batch_dim(full_tree_dict)


    @classmethod
    def make_qeidx(cls, query_idx, edges_lhs):
        return dataset_ops.make_qeidx(query_idx, edges_lhs)


    @classmethod
    def add_semi(cls, element: Element, tree: Tree, config: dict) -> Element:
        '''
            Description

                This method adds the semi edge connecting the "empty"
                node to the root node in the tree. This is active for
                every full tree, but active only in the partial trees
                for those which include the root node in the lhs tree.

                The semi edge connects the root node to itself.

                When the query_idx value is equal to -1, the partial tree is empty.
                In this case, the semi edge is added as [-1, -1] with a mask value
                of zero.
        '''
        semi_dict = cls.make_semi(
            edges=element.edges, edges_mask=element.edges_mask,
            edges_lhs=element.edges_lhs, edges_mask_lhs=element.edges_mask_lhs,
            query_idx=element.query_idx
        )
        return element.update_fields(semi_dict)


    @classmethod
    def add_qeidx(cls, element: Element) -> Element:
        element.qeidx = cls.make_qeidx(element.query_idx, element.edges)
        return element


    @classmethod
    def make_semi(cls, edges, edges_mask, edges_lhs, edges_mask_lhs, query_idx):
        return dataset_ops.make_semi(edges, edges_mask, edges_lhs, edges_mask_lhs, query_idx)


    @classmethod
    def preaug_pad(cls, element: Element, config: dict) -> Element:
        _require_optional_dependency(cv2, "opencv-python")
        seg = element.segmentation
        s = int(seg.shape[0])
        assert s == seg.shape[1]
        pad_size = config['pad_size']
        if pad_size > 0:
            seg = np.pad(
                seg, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                mode='constant', constant_values=config['pad_val']
            )
            element.segmentation = seg
        tgt_seg_size = config['tgt_seg_size']
        if seg.shape[0] != tgt_seg_size or seg.shape[1] != tgt_seg_size:
            new_seg = cv2.resize(
                seg[:, :, 0], (tgt_seg_size, tgt_seg_size),
                interpolation=cv2.INTER_LINEAR
            )
            element.segmentation = new_seg.reshape(tgt_seg_size, tgt_seg_size, 1)
        f = 1.0 / float(s + 2 * pad_size)
        offset = np.array([pad_size, pad_size]).astype(np.float32)
        if element.global_pos is not None:
            element.global_pos = f * (s * element.global_pos + offset)
        if element.global_pos_lhs is not None:
            element.global_pos_lhs = f * (s * element.global_pos_lhs + offset)
        if element.radius is not None:
            element.radius = f * s * element.radius
        if element.radius_lhs is not None:
            element.radius_lhs = f * s * element.radius_lhs
        return element


    @classmethod
    def filter_non_proximal(cls, element: Element, tree: Tree,
                            config: dict) -> Element:
        '''
            Description

                This method sets the node and edges masks to zero in
                the partial tree for all nodes or edges which are not
                proximal to the query node.
        '''
        if (
            element.node_mask_lhs is None
            or element.edges_lhs is None
            or element.edges_mask_lhs is None
            or element.query_idx is None
        ):
            return element

        filter_dict = cls.filter_non_proximal_aux(
            query_idx=element.query_idx, node_mask_lhs=element.node_mask_lhs,
            parents=element.parents, edges_lhs=element.edges_lhs,
            edges_mask_lhs=element.edges_mask_lhs
        )
        return element.update_fields(filter_dict)


    @classmethod
    def filter_non_proximal_aux(cls, query_idx, node_mask_lhs, parents,
                                edges_lhs, edges_mask_lhs):
        return dataset_ops.filter_non_proximal_aux(
            query_idx,
            node_mask_lhs,
            parents,
            edges_lhs,
            edges_mask_lhs,
        )


    @classmethod
    def augment_pos(cls, element: Element, config: dict) -> Element:
        '''
            Description

                The jitter augmentation add random gaussian noise
                to the pos terms.
        '''
        if config['allow_jitter']:
            # only jitter lhs tree
            if element.global_pos_lhs is not None:
                element.global_pos_lhs = cls.do_jitter(
                    element.global_pos_lhs, element.node_mask, config
                )

        return element


    @classmethod
    def do_jitter(cls, pos, mask, config):
        return dataset_ops.do_jitter(pos, mask, config)



    @classmethod
    def make_metadata(cls, tree):
        chunk_index = np.zeros((1), dtype=np.int64)
        chunk_epoch = np.zeros((1), dtype=np.int64)
        if tree.metadata is not None:
            if 'chunk_index' in tree.metadata.keys():
                chunk_index[0] = int(tree.metadata['chunk_index'])
            if 'chunk_epoch' in tree.metadata.keys():
                chunk_epoch[0] = int(tree.metadata['chunk_epoch'])
        return {
            'chunk_index': chunk_index,
            'chunk_epoch': chunk_epoch
        }


    @classmethod
    def save_batch(cls, path, batch_dict):
        if not os.path.exists(path):
            os.makedirs(path)
        for key_str, arr in batch_dict.items():
            arr_path = os.path.join(path, key_str + ".npy")
            save_array(arr_path, arr)


    @classmethod
    def load_batch(cls, path, return_tensor=False, return_cuda=False):
        arr_paths = listdir_fullpath(path)
        batch_dict = {}
        for arr_path in arr_paths:
            arr_name = os.path.basename(arr_path).split('.')[0]
            batch_dict[arr_name] = load_array(arr_path)
            if return_tensor:
                batch_dict[arr_name] = to_tensor(batch_dict[arr_name])
            if return_cuda:
                assert return_tensor
                batch_dict[arr_name] = batch_dict[arr_name].cuda()

        return batch_dict


    def __iter__(self):
        while True:
            element = self.try_put_next()
            if element is not None:
                yield element


    def __len__(self):
        # return arbitrary large number (IterableDataset)
        return 1e7

# --- end public data: dataset.py ---
