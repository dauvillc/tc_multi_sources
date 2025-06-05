import collections.abc
import random
import string


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            if k in d and d[k] is None:
                d[k] = v
            else:
                d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_random_code():
    """Generates a random code containing 2 digits and 6 lowercase letters."""
    chars = random.choices(string.ascii_lowercase, k=6)
    nums = random.choices(string.digits, k=2)
    # Shuffle the characters and numbers
    code = chars + nums
    random.shuffle(code)
    return "".join(code)
