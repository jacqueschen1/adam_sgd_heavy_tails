import argparse


def qparse(descr, args=None, flags=None):
    """Parses the command line call for positional arguments and flags.

    Args:
        descr (str): Main description of the script.
        args: List of ``(name, descr)`` defining positional args.
        flags: List of ``(name, descr)`` defining optional flags.

    Returns:
        args: Namespace object

    Example:
        .. code-block:: python

            args = qparse(
                descr = "My first experiment",
                args = [
                    ("dataset", "The dataset to run on")
                ],
                flags = [
                    ("fig1", "Plot figure 1"),
                    ("fig2", "Plot figure 2"),
                    ("show", "Show plots during execution"),
                    ("save", "Save plots during execution")
                ],
            )

        will result in the following call to ``python myscript.py --help``:

        .. code-block:: console

            usage: myscript.py [-h] [--fig1] [--fig2] [--show] [--save] dataset

            My first experiment

            positional arguments:
              dataset     Dataset to run on

            optional arguments:
              -h, --help  show this help message and exit
              --fig1      Plot figure 1
              --fig2      Plot figure 2
              --show      Show plots during execution
              --save      Save plots during execution

        The arguments and flags will be accessible through
        ``args.dataset``, ``args.fig1``, ``args.fig2``, ``args.show``,
        and ``args.save``.
        Positional arguments are required.
        Flags are ``False`` by default.
    """

    parser = argparse.ArgumentParser(description=descr)
    if flags is not None:
        for flag, help_text in flags:
            parser.add_argument(
                "--" + flag,
                action="store_true",
                help=help_text,
                default=False,
            )
    if args is not None:
        for arg, help_text in args:
            parser.add_argument(
                arg,
                default=None,
                type=str,
                help=help_text,
            )

    return parser.parse_args()
