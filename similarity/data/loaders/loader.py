from abc import abstractmethod

from similarity.data.dataset import Dataset
from similarity.data.text import TextField


class DataLoader:
    @property
    @abstractmethod
    def train(self) -> Dataset:
        """Returns the training data."""
        pass

    @property
    @abstractmethod
    def dev(self) -> Dataset:
        """Returns the validation data."""
        pass

    @property
    @abstractmethod
    def test(self) -> Dataset:
        """Returns the test data."""
        pass

    @property
    @abstractmethod
    def text_field(self) -> TextField:
        """Returns the text field."""
        pass

    def print_stats(self) -> None:
        """Prints statistics about the data."""
        print()
        print(f"Total size = {len(self.train) + len(self.dev) + len(self.test):,}")
        print()
        print(f"Train size = {len(self.train):,}")
        print(f"Dev size = {len(self.dev):,}")
        print(f"Test size = {len(self.test):,}")
        print()
        # print(f'Vocabulary size = {len(self.text_field.vocabulary):,}')
        print()
