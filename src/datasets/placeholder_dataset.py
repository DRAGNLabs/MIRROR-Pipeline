from datasets.mirror_dataset import MirrorDataset


class PlaceholderDataset(MirrorDataset):
    @property
    def dataset_id(self) -> str:
        return 'placeholder_dataset'

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return "this is an example text"
