def trunc_val(val, min_val, max_val):
    val = min(val, max_val)
    val = max(val, min_val)
    return val
