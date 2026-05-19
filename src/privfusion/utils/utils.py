from termcolor import colored


def print_colored_dict(d, prefix=""):
    for key, value in d.items():
        key_str = f"{prefix}{colored(str(key), 'magenta')}: "
        if isinstance(value, dict):
            print(key_str)
            print_colored_dict(value, prefix + "  ")
        else:
            value_str = colored(str(value), "green")
            print(key_str + value_str)


def print_colored_list_of_dicts(list_of_dicts):
    for idx, d in enumerate(list_of_dicts):
        print(f"\n{colored(f'Entry {idx + 1}: ', 'cyan')}")
        print_colored_dict(d)


def print_colored_nested_dict(d):
    for key, value_list in d.items():
        print(f"\n{colored(f'Key: {key}', 'cyan')}")
        for idx, item in enumerate(value_list):
            print(f"  Entry {idx + 1}:")
            print_colored_dict(item)
