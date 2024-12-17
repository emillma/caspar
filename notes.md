# SymCUDA 
It's about time I updated the readme...
There are now two examples, the first one solves the biggest bundle adjustment problem from the BAL dataset (3s on my 4090) and the other is a small illustration showcasing how different types of factors can be used together. Both also showcase how symbolic functions can be trasformed to CODA kernels. 

### TODO
There are a lot of things to add to the solver and tests to write, including
- Equality and inequality constraints
- Better overview of solver options
- Dynamic update of problems (e.g. add points and cameras)
- Verify if it works as indeded when nodes are not used in any factor
- Better use of dynamic dampening based on step quality (currently just a lot slower)
- Write tests with problems that should converge to 0.
- Possibly add dynamic epsilon
- Common factors like IMU preintegration (SE23?)
- Check bug where Nan cause cuda to halt
- Caller for C++
- More efficient expression sorting (solve sub sorting problems)
- Better search in expression sorting
- Support expressions with multiple outputs (to get sin and cos simultaniously)
- Use CUDA graphs and move entire pipeline to CUDA (dynamic parallelism)
- Add use of texture memory for direct SLAM applications and registration
- Automatic use of float64 if float32 does not converge
- Better frontend (don't use strings in kernel parameterization)
- Make the jinja template more readable
- Reduce amount if block syncs (use fence instead of sync when sharing data)
- Add data validation to kernel launcner!!!
- Find better tuning for solver parameters (inner iterations, dampening)
- Implement incremental methods (Use analytic hessian for step tolerance)
- Add preconditioner for band tridiagonal structures
- Share cse between factors when indices are known (e.g. preintegration)
- Add quality of life stuff like getting covariance between nodes  
- Share data between arguments of same type
- Use analytic hessian (Initial test shows bad results)
- Allow inout args (read and write to same arg)
- Make it conform to the torch graph node type (backward pass based on jac at x_min)
- Make tool to evaluate performance gain of fusing/splitting kernels (nCSE, memops)
- Search subspace using prev p_vec instead of line (promising results)

### Maintainability / code quality

### Constraints
- Nullspace
- Augmented Lagrangian

### Incremental solver
- Check if step is large enough to require linealization
- Add later developments from ISAM2 and others

### Register allocation
- Imprement https://arxiv.org/abs/2309.03765
- Implement read write as a functions?
- Add MIMO functions (for sincos, coalesced read/write)

### Launcher performance
- Use streams
- Graph capture 
- Dynamic kernel launch (cuda only solver?)

### PCG performance
- Monitor step quality inside PCG
- First stepp partially conjugent on last final step (less sensitive to inner iterations?)


### Robustness (and speed?)
- Dynamic epsilon
- Float64 + tensor cores (8x8x4 supported)

### Tensor cores
- Float64 (8x8x4)
- __nv_bfloat16 (32x8x16)

### Direct SLAM / registration problems
- Use texture memory
- Custiom *register* symbolic functions?

