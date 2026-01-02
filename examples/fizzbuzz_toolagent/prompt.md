# Task: Implement a FizzBuzz Function

## Objective
Create a Python file called `fizzbuzz.py` that implements the classic FizzBuzz problem.

## Requirements

Write a function called `fizzbuzz(n)` that:
1. Takes an integer `n` as input
2. Returns a list of strings from 1 to n where:
   - Numbers divisible by 3 are replaced with "Fizz"
   - Numbers divisible by 5 are replaced with "Buzz"
   - Numbers divisible by both 3 and 5 are replaced with "FizzBuzz"
   - All other numbers are converted to strings

## Example

```python
fizzbuzz(15)
# Returns: ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]
```

## Instructions

1. Read this prompt to understand the requirements
2. Create the file `fizzbuzz.py` with the implementation
3. Include a `if __name__ == "__main__"` block that prints the result of `fizzbuzz(20)`

## Acceptance Criteria

- Function handles edge cases (n=0, n=1)
- Function returns a list, not prints
- Code is clean and readable
