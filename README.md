# MET

Welcome to the Machine Learning Engineering teams' online technical interview - thanks for taking the time!

This challenge aims to highlight a few different areas of your technical skillset. We'll be looking for:

- Collaborative communication and problem-solving (**soft skills**)
- Proficiency and prowess with your programming environment (**medium skills**)
- The ability to break down and implement a solution to the problem (**hard skills**)

## Challenge

As a part of the Machine Learning Engineering team, you've been tasked to implement a function that can **calculate the determinant of a `n by n` matrix**.

- We've stubbed out a function within the [det.py](./det.py) file.
- We've written some basic unit tests in the [test_det.py](./test_det.py) file.

You can use Google etc to look up what the matrix determinant is and any Python / library API documentation.

**NOTE:** We do not expect you to implement an optimal solution, nor do we expect you to come up with a complete solution within the time frame. Implementing a naive solution that can be easily communicated.

## Hints (only if required)

If you get stuck, here's some extra help to push you in the right direction.

<!-- Hint 1 -->
<details>
<summary>Hint 1</summary>

We like this simple explanation of how to calculate the matrix determinant.
- https://www.mathsisfun.com/algebra/matrix-determinant.html

Can you see how the `3 by 3` relates to the `2 by 2`?

Would a **recursive** algorithm be useful here?

</details>

<!-- Hint 2 -->
<details>
<summary>Hint 2</summary>

We know that a recursive algorithm calls itself. If we're not careful we may continue to call ourselves until our program runs out of memory...

What would our **stopping condition** be? Can we implement this?

</details>

<!-- Hint 3 -->
<details>
<summary>Hint 3</summary>

Let's break down the `3 by 3` case.

We can see that the `3 by 3` case is each value of the first row multiplied by the determinant of the `2 by 2` sub-matrix.

Let's first **iterate** just this first row. How is the position of the current value we're iterating related to the formation of the sub-matrix?

</details>
