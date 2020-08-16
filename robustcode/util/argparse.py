import argparse
import configparser


class ArgConfigParser(argparse.ArgumentParser):
    """
    Extended version of ArgumentParser that optionally reads default values from a config file.
    Values provided in the command line override those in the config file.
    """

    def __init__(self, *args, **kwargs):
        super(ArgConfigParser, self).__init__(
            *args, **kwargs, conflict_handler="resolve"
        )
        self.add_argument(
            "--config", default=None, help="Specify config file(s)", nargs="*"
        )

    def parse_args(self, args=None, namespace=None):
        # First parse an optional config file
        parser = argparse.ArgumentParser("Config Parser", add_help=False)
        parser.add_argument("-c", "--config", help="Specify config file(s)", nargs="*")

        args, remaining_argv = parser.parse_known_args()

        if args.config:
            for config_file in args.config:
                # Read all values from the config file
                config_args = self.get_args_from_config(config_file)
                # Prepend the values from config before values from command line
                # This ensures that values from command line have precedence
                remaining_argv = (
                    config_args + remaining_argv + ["--config", args.config]
                )

        return super().parse_args(remaining_argv, namespace)

    def get_args_from_config(self, config_file: str):
        config = configparser.ConfigParser()
        config.read(config_file)

        config_args = []
        for key, value in config.defaults().items():
            config_args.append("--{}".format(key))
            config_args.append(value)
        return config_args
