import numpy as np

def print_stats_np(arr, name: str) -> None:
    print_p("np array: {0} - {1} - {2} - minmax=({3}, {4}), mean={5}\n".format(
        name,
        arr.shape,
        arr.dtype,
        round(np.min(arr), 2),
        round(np.max(arr), 2),
        round(np.mean(arr), 3)))


def print_p(text, end="\n"):
    """ Print function for on Peregrine. It needs a flush before printing. """
    print(text, flush=True, end=end)


def print_(iets):
    print(f"{iets}", flush=True)