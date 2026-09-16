import pandas as pd
from src.dataloader import SplitDataset
from sklearn.model_selection import train_test_split


class SplitGenerator:
    def generate(
        self,
        dataset: SplitDataset,
        seed: int,
    ) -> SplitDataset:
        development_set = pd.concat(
            [
                dataset.train,
                dataset.validation,
            ],
            ignore_index=True,
        )

        train_df, validation_df = train_test_split(
            development_set,
            test_size=len(dataset.validation) / len(development_set),
            random_state=seed,
        )

        return SplitDataset(
            train=train_df,
            validation=validation_df,
            test=dataset.test,
        )
