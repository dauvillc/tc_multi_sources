from string import Template


class DeltaTemplate(Template):
    delimiter = "%"


def strfdelta(tdelta, fmt):
    """Formats, a timedelta object into a string.
    Taken from https://stackoverflow.com/questions/8906926/formatting-timedelta-objects/8907269#8907269
    thx to Shawn Chin and Peter Mortensen.
    """
    d = {"D": tdelta.days}
    d["H"], rem = divmod(tdelta.seconds, 3600)
    d["M"], d["S"] = divmod(rem, 60)
    t = DeltaTemplate(fmt)
    return t.substitute(**d)


def format_tdelta(tdelta, seconds=False):
    """Formats a timedelta object in a human-readable format."""
    days = tdelta.days
    hours, rem = divmod(tdelta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if seconds:
        return f"{days}d{hours}h{minutes}m{seconds}s"
    return f"{days}d{hours}h{minutes}m"
