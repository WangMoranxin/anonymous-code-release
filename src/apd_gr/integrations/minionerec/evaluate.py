"""MiniOneRec uses exact equality over its three semantic SID tokens."""


def exact_sid_match(prediction, target) -> bool:
    return tuple(prediction) == tuple(target)
