from mymodule import my_func


def separate_dataset(dataset, train_ratio=0.8):
    """
    Splits the dataset into training and testing sets based on the specified ratio.

    Parameters:
    dataset (list): The dataset to be split.
    train_ratio (float): The proportion of the dataset to include in the training set.

    Returns:
    tuple: A tuple containing the training and testing datasets.
    """
    train_size = int(len(dataset) * train_ratio)
    train_set = dataset[:train_size]
    test_set = dataset[train_size:]

    return train_set, test_set
