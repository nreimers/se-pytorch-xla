import gzip
import json
import random


class RedditDataset:
    """
    A class that handles the reddit data files
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        while True:
            with gzip.open(self.filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)

                    if "response" in data and "context" in data:
                        yield [data["response"], data["context"]]


class Dataset:
    """
    A class that handles one dataset
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        max_dataset_size = 20 * 1000 * 1000  # Cache small datasets in memory
        dataset = []
        data_format = None

        while dataset is None or len(dataset) == 0:
            with gzip.open(self.filepath, "rt") as fIn:
                for line in fIn:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        data = data['texts']

                    if data_format is None:
                        data_format = len(data)

                    # Ensure that all entries are of the same 2/3 col format
                    assert len(data) == data_format

                    if dataset is not None:
                        dataset.append(data)
                        if len(dataset) >= max_dataset_size:
                            dataset = None

                    yield data

        # Data loaded. Now stream to the queue
        # Shuffle for each epoch
        while True:
            random.shuffle(dataset)
            for data in dataset:
                yield data
