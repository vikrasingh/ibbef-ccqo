# ibbef-ccqo
This package provides the testing and algorithm files to solve the following **Cardinality Constrained Quadratic Optimization (CCQO)** problem
```math
\min_{ x \in R^n} \,\, \frac{1}{2}x^{t}Ax+b^{t}x+c \,\, \text{   subject to   } \,\, \|x\|_{0}\leq k 
``` 
where $A \in R^{n \times n}$ is a positive semidefinite matrix, $b \in R^n$, $c \in R$, and $`\|\cdot\|_{0}`$ is a pseudo-norm which counts the no. of non-zero entries of $`x`$.
