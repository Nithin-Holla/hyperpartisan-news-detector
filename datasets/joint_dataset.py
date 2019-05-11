import torch.utils.data as data


class JointDataset(data.Dataset):
    def __init__(self, *datasets):
        self._datasets = datasets
        self._data_size = max([dataset.__len__() for dataset in datasets])

    def __getitem__(self, idx):
        result_list = []

        for dataset in self._datasets:
            # get id according to the current dataset length
            # so if one of the datasets is smaller, we don't call invalid id
            dataset_idx = idx % dataset.__len__()
            
            item = dataset.__getitem__(dataset_idx)
            result_list.append(item)

        return result_list


    def __len__(self):
        return self._data_size