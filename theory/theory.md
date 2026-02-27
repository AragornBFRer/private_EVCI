```latex
\section{Setup}\label{sec:setup}

\subsection*{Problem formulation}\label{subsec:problem}
Consider an error-in-variable conformal inference problem where the true covariates (or diagnostic labels) are proxied by a model. Let $Y \in \mathbb{R}$ be the response variable. We do not explicitly model other covariates $X$. Below we consider the binary case:
\begin{itemize}
    \item $Z \in \{0, 1\}$: The gold-standard diagnostic label (the true label).
    \item $\hat{Z} \in \{0, 1\}$: A proxy diagnostic label obtained from a potentially noisy model (e.g., an LLM).
\end{itemize}

During \emph{training}, \emph{calibration}, we only observe $\{(\hat{Z}_i, Y_i)\}_{i=1}^n$. The true label $Z$ is unobserved. During \emph{testing}, for new observations, we obtain the gold-standard $Z_{n+1}$ and want to predict $Y_{n+1}$.
Let $h(Z)$ be the given regression function. We define two types of residuals:
\begin{itemize}
    \item $R = Y - h(Z)$: The true residual.
    \item $\tilde{R} = Y - h(\hat{Z})$: The observable residual based on the proxy diagnosis.
\end{itemize}

Our primary goal is to infer the conditional distribution of the true residual, $P(R \mid Z)$, from the observable distribution $P(\tilde{R} \mid \hat{Z})$ in order to construct valid conformal prediction intervals.



%
%
%
%
%

%
%
%
%
%
\vspace{1em}
\begin{assumption}[Known error mechanism]\label{assump:mechanism}
The error mechanism $P(\hat{Z} \mid Z)$ is known.
\end{assumption}

\begin{assumption}[Known marginals]\label{assump:marginals}
The marginal distribution of the true label is known. Let $\pi_1 = P(Z=1)$ and $\pi_0 = P(Z=0)$. Consequently, the joint distribution $P(\hat{Z}, Z)$ is fully specified.
\end{assumption}

\begin{assumption}[Conditional independence]\label{assump:independence}
The response $Y$ is conditionally independent of the proxy label $\hat{Z}$ given the true label $Z$, i.e., $Y \perp \hat{Z} \mid Z$.
\end{assumption}



%
%
%
%
%

%
%
%
%
%
\vspace{1em}
Under Assumption \ref{assump:independence}, $P(Y \mid \hat{Z}, Z) = P(Y \mid Z)$. However, the observed residual $\tilde{R} = Y - h(\hat{Z})$ may be different from ${R} = Y - h({Z})$ whenever $\hat{Z} \neq Z$. Let $\delta = h(1) - h(0)$ denote the difference in the regression function between the two classes. 
\begin{itemize}
    \item If $\hat{Z} = 1$ and $Z = 0$, then $Y - h(\hat{Z}) = Y - h(1) = Y - h(0) - \delta = R - \delta$.
    \item If $\hat{Z} = 0$ and $Z = 1$, then $Y - h(\hat{Z}) = Y - h(0) = Y - h(1) + \delta = R + \delta$.
\end{itemize}

This indicates that $P(Y - h(\hat{Z}) \mid \hat{Z}=1, Z=0)$ is equivalent to evaluating $P(R = \tilde{R} + \delta \mid Z=0)$. Let $f_z(r) = P(R = r \mid Z=z)$ denote the target conditional densities of the true residual. Then 
\begin{align}
    P(\tilde{R} = r, \hat{Z} = 1) &= P(\hat{Z}=1, Z=1) f_1(r) + P(\hat{Z}=1, Z=0) f_0(r + \delta) \\
    P(\tilde{R} = r, \hat{Z} = 0) &= P(\hat{Z}=0, Z=1) f_1(r - \delta) + P(\hat{Z}=0, Z=0) f_0(r)
\end{align}

Let $M$ be the joint probability matrix of the labels:
\begin{equation}
    M = \begin{pmatrix}
    P(\hat{Z}=1, Z=1) & P(\hat{Z}=1, Z=0) \\
    P(\hat{Z}=0, Z=1) & P(\hat{Z}=0, Z=0)
    \end{pmatrix}
    = 
    \begin{pmatrix}
    \pi_1 P(\hat{Z}=1 \mid Z=1) & \pi_0 P(\hat{Z}=1 \mid Z=0) \\
    \pi_1 P(\hat{Z}=0 \mid Z=1) & \pi_0 P(\hat{Z}=0 \mid Z=0)
    \end{pmatrix}
\end{equation}

\begin{proposition}\label{prop:inverse}
Given the joint distribution of the proxy residuals $P(\tilde{R}=r, \hat{Z}=z)$, the target true residual distributions $f_1(r)$ and $f_0(r)$ can be recovered via the following matrix inversion:
\begin{equation}
    \begin{pmatrix} f_1(r) \\ f_0(r) \end{pmatrix} 
    = M^{-1} \left[ \begin{pmatrix} P(\tilde{R}=r, \hat{Z}=1) \\ P(\tilde{R}=r, \hat{Z}=0) \end{pmatrix} - V(r) \right]
\end{equation}
where $V(r)$ is an adjustment vector defined as:
\begin{equation}
    V(r) = \begin{pmatrix} 
    P(\hat{Z}=1, Z=0) \big[ f_0(r+\delta) - f_0(r) \big] \\ 
    P(\hat{Z}=0, Z=1) \big[ f_1(r-\delta) - f_1(r) \big] 
    \end{pmatrix}
\end{equation}
\end{proposition}

\begin{proof}[Proof of Proposition~\ref{prop:inverse}]
By expanding the matrix multiplication $M \begin{pmatrix} f_1(r) \\ f_0(r) \end{pmatrix} + V(r)$, we have:
\begin{align*}
    &\begin{pmatrix} 
    P(\hat{Z}=1, Z=1) f_1(r) + P(\hat{Z}=1, Z=0) f_0(r) \\
    P(\hat{Z}=0, Z=1) f_1(r) + P(\hat{Z}=0, Z=0) f_0(r) 
    \end{pmatrix}
    +
    \begin{pmatrix} 
    P(\hat{Z}=1, Z=0) \big[ f_0(r+\delta) - f_0(r) \big] \\ 
    P(\hat{Z}=0, Z=1) \big[ f_1(r-\delta) - f_1(r) \big] 
    \end{pmatrix} \\[3px]
    &= \begin{pmatrix} 
    P(\hat{Z}=1, Z=1) f_1(r) + P(\hat{Z}=1, Z=0) f_0(r+\delta) \\
    P(\hat{Z}=0, Z=1) f_1(r-\delta) + P(\hat{Z}=0, Z=0) f_0(r) 
    \end{pmatrix}
\end{align*}
This is equivalent to the vector of joint probabilities $\begin{pmatrix} P(\tilde{R}=r, \hat{Z}=1) \\ P(\tilde{R}=r, \hat{Z}=0) \end{pmatrix}$ as derived previously. Left-multiplying by $M^{-1}$ yields the desired expression.
\end{proof}
```