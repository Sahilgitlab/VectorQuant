"""
Regime Detection
"""
import math
from vectorquant.core.probability import normal_pdf

def forward_algorithm_hmm(observations, initial_probs, trans_matrix, emission_params):
    """
    Computes P(S_t | S_{t-1}) and overall likelihood of observation sequence.
    Given discrete hidden states and Gaussian emission probabilities.
    initial_probs: [P(S_1), P(S_2), ...]
    trans_matrix: [[P(S_1|S_1), P(S_2|S_1)], ... ] (P[i][j] = prob moving i -> j)
    emission_params: [(mu_1, sigma_1), (mu_2, sigma_2), ...]
    Returns list of alpha vectors (probabilities of states at each step).
    """
    if not observations: return []
    
    n_states = len(initial_probs)
    alphas = []
    
    # Initialize alpha_1
    alpha_1 = []
    x1 = observations[0]
    for i in range(n_states):
        mu, sigma = emission_params[i]
        prob_emit = normal_pdf(x1, mu, sigma)
        alpha_1.append(initial_probs[i] * prob_emit)
        
    alphas.append(alpha_1)
    
    # Forward recursion
    for t in range(1, len(observations)):
        xt = observations[t]
        alpha_t = []
        for j in range(n_states):
            mu, sigma = emission_params[j]
            prob_emit = normal_pdf(xt, mu, sigma)
            
            # sum over i: alpha_{t-1}[i] * A_{ij}
            sum_transitions = sum(alphas[t-1][i] * trans_matrix[i][j] for i in range(n_states))
            alpha_t.append(prob_emit * sum_transitions)
            
        alphas.append(alpha_t)
        
    return alphas

def viterbi_algorithm_hmm(observations, initial_probs, trans_matrix, emission_params):
    """
    Finds the most likely sequence of hidden states.
    """
    if not observations: return []
    
    n_states = len(initial_probs)
    T = len(observations)
    
    viterbi = [[0.0 for _ in range(n_states)] for _ in range(T)]
    backpointer = [[0 for _ in range(n_states)] for _ in range(T)]
    
    # Init
    x1 = observations[0]
    for s in range(n_states):
        mu, sigma = emission_params[s]
        # Use log probs to prevent underflow
        prob_emit = normal_pdf(x1, mu, sigma)
        if prob_emit > 1e-15 and initial_probs[s] > 1e-15:
            viterbi[0][s] = math.log(initial_probs[s]) + math.log(prob_emit)
        else:
            viterbi[0][s] = float('-inf')
            
    # Recursion
    for t in range(1, T):
        xt = observations[t]
        for s in range(n_states):
            mu, sigma = emission_params[s]
            prob_emit = normal_pdf(xt, mu, sigma)
            log_emit = math.log(prob_emit) if prob_emit > 1e-15 else float('-inf')
            
            max_tr_prob = float('-inf')
            best_prev_state = 0
            
            for prev_s in range(n_states):
                tr_prob = trans_matrix[prev_s][s]
                log_tr = math.log(tr_prob) if tr_prob > 1e-15 else float('-inf')
                
                prob = viterbi[t-1][prev_s] + log_tr
                if prob > max_tr_prob:
                    max_tr_prob = prob
                    best_prev_state = prev_s
                    
            viterbi[t][s] = max_tr_prob + log_emit
            backpointer[t][s] = best_prev_state
            
    # Terminate
    best_last_state = 0
    max_prob = float('-inf')
    for s in range(n_states):
        if viterbi[T-1][s] > max_prob:
            max_prob = viterbi[T-1][s]
            best_last_state = s
            
    # Backtrack
    best_path = [best_last_state]
    for t in range(T-1, 0, -1):
        best_path.insert(0, backpointer[t][best_path[0]])
        
    return best_path

def cusum(time_series, target_mean=0.0, drift=0.0, threshold=5.0):
    """
    Cumulative Sum (CUSUM) change point detection.
    Detects shifts in the mean level of a time series.
    Returns indices where a change point is signalled.
    """
    positive_cusum = [0.0]
    negative_cusum = [0.0]
    change_points = []
    
    for i in range(1, len(time_series)):
        x = time_series[i]
        
        # S_i^+ = max(0, S_{i-1}^+ + X_i - target - drift)
        sp = max(0.0, positive_cusum[-1] + x - target_mean - drift)
        positive_cusum.append(sp)
        
        # S_i^- = max(0, S_{i-1}^- - X_i + target - drift)
        sn = max(0.0, negative_cusum[-1] - x + target_mean - drift)
        negative_cusum.append(sn)
        
        if sp > threshold or sn > threshold:
            change_points.append(i)
            # Optional: reset after detection
            positive_cusum[-1] = 0.0
            negative_cusum[-1] = 0.0
            
    return change_points
