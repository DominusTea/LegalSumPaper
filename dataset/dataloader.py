from torch.utils.data import DataLoader
import torch

class CustomDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(CustomDataloader, self).__init__(*args, **kwargs)
    def collate_fn(self, batch):
        data = [item[0] for item in batch] # data is text
        target = [item[1] for item in batch] # target is also text (summary)
        target = torch.LongTensor(target)
        return [data, target]

class CustomDataloaderWithTags(DataLoader):
    '''
    Should work in every case.
    Metadata must be placed after the first 2 elements in each item. therefore each batch
    will take the form [[text, summary, [metadata1, ...,metadatan], ..., [text, summary, [metadata1, ...,metadatan]]

    '''
    def __init__(self, *args, **kwargs):
        super(CustomDataloaderWithTags, self).__init__(*args, **kwargs)
    def collate_fn(self, batch):
        data = [item[0] for item in batch] # data is text
        target = [item[1] for item in batch] # target is also text (summary)
        target = torch.LongTensor(target)
        tag = [item[2:] for item in batch]
        return [data, target, tag]

class TransformersDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(TransformersDataloader, self).__init__(*args, **kwargs)
        if 'tokenizer' in kwargs:
            self.tokenizer = kwargs['tokenizer']
    def collate_fn(self, batch):
        # encode data and target to huggingface form ([CLS], huggingface tokenizer)
        data = [self.tokenizer(item[0])['input_ids'] for item in batch]
        target = torch.LongTensor([self.tokenizer(item[1])['input_ids'] for item in batch])
        return [data, target]
