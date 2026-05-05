
def validate_input(features):
    if len(features) != 9:
        raise ValueError("Input must contain exactly 9 features.")