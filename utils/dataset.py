import numpy as np

def dataset_splitter(dataset, split_percs, seed=42):
    '''
    Splits dataset according to split_percs (which follow the template train%_dev%_test% or train%_test%)
    '''
    # get each subset's length percentage
    percs = split_percs.split('_')
    # make them floats
    percs=np.array(percs,dtype=float)
    # if any of the percentages is >=1.0 then divide every percentage by 100
    if any(perc >= 1.0 for perc in percs):
        percs /= 100

    # if len(percs) == 2:
    #     _,test_dataset = dataset.split_dataset(lengths=percs, seed=seed)
    # elif len(percs) ==3:
    #     _, _, test_dataset = dataset.split_dataset(lengths=percs, seed=seed)
    # else:
    if len(percs) not in [2,3]:
        raise ValueError("split percs is not of the form: [train_perc_dev_perc_test_perc | train_perc_test_perc]")
    assert sum(percs) == 1.0, "Percentages must add up to 1.0"

    # calculate lengths
    N = len(dataset)
    subset_lengths=[round(N*perc) for perc in percs[:-1]]
    # add last subset's length seperately to assert all lengths add up to N
    subset_lengths += [N - sum(subset_lengths)]
    print('Split datasets to percentages', percs, " and corresponding lengths", subset_lengths)

    # get the split that corresponds to the test subset. This should always be the last
    test_dataset = dataset.split_dataset(lengths=subset_lengths, seed=seed)


    return test_dataset
