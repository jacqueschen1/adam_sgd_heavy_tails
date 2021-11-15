import functools
import pdb
import data_helpers as h


def _SELECT_SGD(df):
    return (df[h.OPT_NAME] == "SGD") & (
        (df[h.OPT_MOMENTUM].isnull()) | (df[h.OPT_MOMENTUM] == 0)
    )


def _SELECT_SGD_M(df):
    return (df[h.OPT_NAME] == "SGD") & (df[h.OPT_MOMENTUM] == 0.9)


def _SELECT_ADAM_NM(df):
    return (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0)


def _SELECT_ADAM(df):
    return (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0.9)


def _SELECT_PROBLEM(df, dataset, model, batch_size, acc_step):
    SELECT_PATTERN = (df[h.DATASET] == dataset) & (df[h.MODEL] == model)
    if batch_size > 0:
        SELECT_PATTERN = SELECT_PATTERN & (df[h.BATCH_SIZE] == batch_size)
    if acc_step > 0:
        SELECT_PATTERN = SELECT_PATTERN & (df[h.ACC_STEP] == acc_step)
    return SELECT_PATTERN


def _filter_tags(df):
    for tag in h.tags_to_checkfuncs.keys():
        df = df[~df["tags"].apply(lambda tags: tag in tags)]

    manually_set_tags = ["old", "old_full_batch_runs"]
    for tag in manually_set_tags:
        df = df[~df["tags"].apply(lambda tags: tag in tags)]

    return df


def _filter_out_crashed_runs(df):
    """Filters out runs that ran out of time, memory, or threw an exception."""
    return df[df["state"] == "finished"]


def _filter_out_unfinished_runs(df):
    return df[(df[h.EPOCH] == df[h.MAX_EPOCH] - 1)]


def _get_optim_results(df):
    ADAM = (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0.9)
    ADAM_NM = (df[h.OPT_NAME] == "Adam") & (df[h.OPT_B1] == 0)
    SGD = (df[h.OPT_NAME] == "SGD") & (
        (df[h.OPT_MOMENTUM].isnull()) | (df[h.OPT_MOMENTUM] == 0)
    )
    SGD_M = (df[h.OPT_NAME] == "SGD") & (df[h.OPT_MOMENTUM] == 0.9)
    return df[ADAM], df[SGD], df[ADAM_NM], df[SGD_M]


def _select_results(df, dataset, model, batch_size, acc_step, big_batch, full_batch):
    SELECT = _SELECT_PROBLEM(df, dataset, model, batch_size, acc_step)

    if big_batch and not full_batch:
        SELECT = SELECT & (df[h.DROP_LAST] == True)
    elif full_batch and not big_batch:
        SELECT = SELECT & (df[h.FULL_BATCH])
    elif not full_batch and not big_batch:
        SELECT = (
            SELECT
            & df[h.DROP_LAST].isnull()
            & ((df[h.FULL_BATCH] == False) | df[h.FULL_BATCH].isnull())
        )
    else:
        raise ValueError(
            f"Unsupported combination of big batch ({big_batch}) and full batch ({full_batch})."
        )

    return df[SELECT]


def _filter_out_histogram_experiments(df):
    df = df[~(df["trained_norms"] == True)]
    df = df[~(df["init_noise_norm"] == True)]
    df = df[~(df["noise_norm_train"] == True)]
    return df


def _take_only_the_last_run_for_each_hash(df):
    reduced_df = df[["hash", "_timestamp"]]
    reduced_df = reduced_df.groupby("hash").max().reset_index()
    df = reduced_df.merge(
        df, how="inner", left_on=["hash", "_timestamp"], right_on=["hash", "_timestamp"]
    )
    return df


def _run_all_filters(df):
    df = _filter_tags(df)
    df = _filter_out_crashed_runs(df)
    df = _filter_out_histogram_experiments(df)
    df = _take_only_the_last_run_for_each_hash(df)
    return df


@functools.lru_cache()
def get_filtered_data():
    df = h.get_data()
    df = _run_all_filters(df)
    return df


def find_step_size_range(
    acc_step, batch_size, big_batch, dataset, full_batch, m_, model
):
    df = get_filtered_data()

    max_ss = df[h.K_SS].max()
    min_ss = df[h.K_SS].min()

    big_batch, full_batch = process_full_batch_input(full_batch, dataset, big_batch)
    df = _select_results(
        df, dataset, model, batch_size, acc_step, big_batch, full_batch
    )

    unique_ss = list(df[h.K_SS].dropna().unique())

    return max_ss, min_ss, unique_ss


def select_runs(acc_step, batch_size, big_batch, dataset, full_batch, m_, model):
    df = get_filtered_data()
    big_batch, full_batch = process_full_batch_input(full_batch, dataset, big_batch)

    df = _select_results(
        df, dataset, model, batch_size, acc_step, big_batch, full_batch
    )
    return _get_optim_results(df)


def process_full_batch_input(full_batch, dataset, big_batch):
    big_batch = big_batch
    if full_batch and dataset in ["wikitext2", "ptb", "mnist"]:
        full_batch = False
        big_batch = True
    return big_batch, full_batch
