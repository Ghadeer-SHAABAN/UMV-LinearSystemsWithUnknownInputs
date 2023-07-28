# UMV-LinearSystemsWithUnknownInputs
This repository offers a straightforward implementation of the Unbiased minimum-variance (UMV) input and state estimation for linear discrete-time systems with and without direct feedthrough.

The code was developed from scratch due to the lack of existing straightforward implementations available online. It serves as a valuable resource for those in search of a simple UMV that may not be readily accessible elsewhere on the internet (to the best of the author's knowledge).

Here I add the links to the original two papers for the mathematics behind and the algorithm explaination. 

Without direct feedthrough [1]: https://www.sciencedirect.com/science/article/abs/pii/S0005109806003189

With direct feedthrough [2]: https://www.sciencedirect.com/science/article/abs/pii/S0005109807000222



## Unbiased minimum-variance input and state estimation for linear discrete-time systems [1]
### The considered discrete time linear model
Consider the linear discrete-time system
$$x_{k+1}=A_k x_k+G_k d_k+w_k,$$
$$y_k=C_k x_k+v_k,$$

where $x_k \in \mathbb{R}^n$ is the state vector, $d_k \in \mathbb{R}^m$ is an unknown input vector, and $y_k \in \mathbb{R}^p$ is the measurement. The process noise $w_k \in \mathbb{R}^n$ and the measurement noise $v_k \in \mathbb{R}^p$ are assumed to be mutually uncorrelated, zero-mean, white random signals with known covariance matrices, $Q_k=\mathbb{E}\left[w_k w_k^{\top}\right]$ and $R_k=\mathbb{E}\left[v_k v_k^{\top}\right]$, respectively. $\left(C_k, A_k\right)$ is observable and  $x_0$ is independent of $v_k$ and $w_k$ for all $k$. Also, we assume that the following sufficient condition for the existence of an unbiased state estimator is satisfied.
$$rank C_k G_{k-1}=rank G_{k-1}=m, \forall k$$

### Implementation and simulation results. 
Run the file: UMV_withoutDirectFeedthrough.py

## Unbiased minimum-variance input and state estimation for linear discrete-time systems with direct feedthrough [2]
### The considered discrete time linear model
Consider the linear discrete-time system
$$x_{k+1} =A_k x_k+G_k d_k+w_k,$$
$$y_k =C_k x_k+H_k d_k+v_k,$$
where $x_k \in \mathbb{R}^n$ is the state vector, $d_k \in \mathbb{R}^m$ is an unknown input vector, and $y_k \in \mathbb{R}^p$ is the measurement. The process noise $w_k \in \mathbb{R}^n$ and the measurement noise $v_k \in \mathbb{R}^p$ are assumed to be mutually uncorrelated, zero-mean, white random signals with known covariance matrices, $Q_k=\mathbb{E}\left[w_k w_k^{\mathrm{T}}\right] \geqslant 0$ and $R_k=$ $\mathbb{E}\left[v_k v_k^{\mathrm{T}}\right]>0$, respectively. The matrices $A_k, G_k, C_k$ and $H_k$ are known and it is assumed that rank $H_k=m$. Throughout the paper, we assume that $\left(A_k, C_k\right)$ is observable and that $x_0$ is independent of $v_k$ and $w_k$ for all $k$. Also, we assume that an unbiased estimate $\hat{x}_0$ of the initial state $x_0$ is available with covariance matrix $P_0^x$.

### Implementation and simulation results. 

Run the file: UMV_withDirectFeedthrough.py


## References:
[1] S. Gillijns and B. De Moor, “Unbiased minimum-variance input and state estimation for linear discrete-time systems,” Automatica, vol. 43, no. 1, pp. 111–116, Jan. 2007, doi: 10.1016/j.automatica.2006.08.002.

[2] S. Gillijns and B. De Moor, “Unbiased minimum-variance input and state estimation for linear discrete-time systems with direct feedthrough,” Automatica, vol. 43, no. 5, pp. 934–937, May 2007, doi: 10.1016/j.automatica.2006.11.016.
