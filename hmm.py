import numpy as np
def forward(V, a, b, initial_distribution):
    c=np.zeros(V.shape[0])
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]
    scaling=np.sum(alpha[0,:])
    alpha[0,:]=alpha[0,:]/scaling
    c[0]=scaling
    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]
        scaling=0
        scaling=np.sum(alpha[t,:])
        alpha[t,:]=alpha[t,:]/scaling
        c[t]=scaling
    eval_prob=-np.sum(np.log(c))
    return alpha , eval_prob
def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])
        sc=0
        sc=np.sum(beta[t,:])
        beta[t,:]=beta[t,:]/sc
    return beta
def baum_welch(V, atrans, btrans, initial_distribution, n_iter=100):
    a=atrans
    b=btrans
    M = a.shape[0]
    init__dist=initial_distribution
    for n in range(n_iter):
            T=V.shape[0]
            xi = np.zeros((M, M, T - 1))
            alpha,ep = forward(V, a, b, initial_distribution)
            beta = backward(V, a, b)
            for t in range(T - 1):
                denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
                for i in range(M):
                    numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                    xi[i, :, t] = numerator / denominator
            gamma = np.sum(xi, axis=1)
            a= np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))
            gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))
            init__dist=gamma[:,0]
            K = b.shape[1]
            denominator = np.sum(gamma, axis=1)
            for l in range(K):
                b[:, l] = np.sum(gamma[:, V == l], axis=1)
            b = np.divide(b, denominator.reshape((-1, 1)),where=denominator!=0)

    return (a, b , init__dist)
def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]
    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution,where=initial_distribution>0 )+np.log(b[:, V[0]],where=b[:, V[0]]>0)
    prev = np.zeros((T - 1, M))
    maxprob=np.zeros((M))
    for t in range(1, T):
        for j in range(M):
            probability = omega[t - 1] + np.log(a[:, j],where=a[:, j]>0) + np.log(b[j, V[t]],where=b[j, V[t]]>0)
            prev[t - 1, j] = np.argmax(probability)
            omega[t, j] = np.max(probability)
    S = np.zeros(T)
    last_state = np.argmax(omega[T - 1, :])
    S[0] = last_state
    maxprob=np.max(omega[-1,:])
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
    S = np.flip(S, axis=0)
    return S,maxprob

