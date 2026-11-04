import os.path as osp

from utils.iotools import mkdir_if_missing
from .bases import BaseImageDataset
from timm.data.random_erasing import RandomErasing
from .market1501 import Market1501
from .msmt17 import MSMT17
from .dukemtmcreid import DukeMTMCreID
from .cuhk03 import CUHK03
datasets = [
    'cuhk03',
    'msmt17',
    'dukemtmc',
    'market1501',
]

class ALL(BaseImageDataset): 
    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
      __factory = {
        'cuhk03': CUHK03,
        'market1501': Market1501,
        'msmt17': MSMT17,
        'dukemtmc': DukeMTMCreID,
      }
      train = []
      query = []
      gallery = []
      number = 0
      for name in datasets:
        self.dataset = __factory[name](root = root, pid_begin = number)
        # self.add_pids("train", len(train))
        # self.add_pids("query", len(query))
        # self.add_pids("gallery", len(gallery))
        number += self.dataset.num_train_pids
        train += (self.dataset.train)
        query += (self.dataset.query)
        gallery += (self.dataset.gallery)

      if verbose:
            print("=> ALL datasets loaded")
            self.print_dataset_statistics(train, query, gallery)
    
      self.train = train
      self.query = query
      self.gallery = gallery
      self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
      self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
      self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def add_pids(self, name, num):
       if name == "train":
          for index in range(0, len(self.dataset.train)):
             self.dataset.train[index][1] += num
       elif name == "query":
          for index in range(0, len(self.dataset.query)):
             self.dataset.query[index][1] += num
       else:
          for index in range(0, len(self.dataset.gallery)):
             self.dataset.gallery[index][1] += num
        


