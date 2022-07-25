import h5py
import lmdb
import pickle as pkl
import numpy as np

SPECIAL_TOKEN_CLIP={"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "<|maskoftext|>", "UNK_TOKEN": "<|unkoftext|>", "PAD_TOKEN": "<|padoftext|>"}



class H5Feature(object):
    def __init__(self, h5_paths, feature_framerate=1.0, max_frames=48):
        self.h5_paths = h5_paths
        self.max_frames = max_frames
        if isinstance(h5_paths, str):
            self.h5_paths = [h5_paths]
        self.h5s = []
        self._get_dim()
    
    def get_feature(self, key):
        if self.h5s == []:
            for h5_path in self.h5_paths:
                self.h5s.append(h5py.File(h5_path, 'r'))
        
        fts = []
        for i, h5 in enumerate(self.h5s):
            if key not in h5:
                fts.append(np.zeros((1, self.dims[i]), dtype=np.float32))
                print(f'Feature No.{i} not found. Video name: {key}')
                print('Use zero instead')
            else:
                fts.append(np.array(h5[key], dtype=np.float32))
        try:
            ft = np.concatenate(fts, axis=-1)
        except ValueError:
            fts = self.align_len(fts)
            ft = np.concatenate(fts, axis=-1)
        ft = ft[:self.max_frames]
        return ft
    
    def __getitem__(self, key):
        return self.get_feature(key)
    
    def align_len(self, fts):
        max_len = max([fts[i].shape[0] for i in range(len(fts))])
        for i in range(len(fts)):
            align_index = np.round(np.linspace(0, fts[i].shape[0] - 1, max_len)).astype('int32')
            fts[i] = fts[i][align_index]
        return fts
    
    def _get_dim(self):
        self.dim = 0
        self.dims = []
        for path in self.h5_paths:
            with h5py.File(path) as h5:
                for key, feature in h5.items():
                    self.dim += feature.shape[1]
                    self.dims.append(feature.shape[1])
                    break
