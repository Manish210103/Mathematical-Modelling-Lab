import math

def sqrt(x):
    """Square root using Newton-Raphson method"""
    if x < 0:
        return float('nan')
    if x == 0:
        return 0
    guess = x / 2.0
    for _ in range(50):
        guess = (guess + x / guess) / 2.0
    return guess

def mean(values):
    """Calculate arithmetic mean"""
    if not values:
        return 0.0
    return sum(values) / len(values)

def variance(values):
    """Calculate variance"""
    if not values:
        return 0.0
    mu = mean(values)
    return sum((x - mu) ** 2 for x in values) / len(values)

def std_deviation(values):
    """Calculate standard deviation"""
    return sqrt(variance(values))
