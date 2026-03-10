"""
Model Calibration Engine
"""
import math

def simple_gradient_descent_calibration(model_func, market_prices, initial_params, learning_rate=0.01, max_iter=1000, tol=1e-5):
    """
    Minimizes Sum(Model(theta) - Market)^2 using finite difference gradients.
    model_func: takes list of params, returns list of prices matching market_prices length.
    """
    params = list(initial_params)
    n_params = len(params)
    h = 1e-5
    
    def loss(p):
        preds = model_func(p)
        return sum((preds[i] - market_prices[i])**2 for i in range(len(preds)))
        
    for _ in range(max_iter):
        current_loss = loss(params)
        if current_loss < tol:
            break
            
        gradients = []
        for i in range(n_params):
            p_plus = list(params)
            p_plus[i] += h
            loss_plus = loss(p_plus)
            
            p_minus = list(params)
            p_minus[i] -= h
            loss_minus = loss(p_minus)
            
            grad = (loss_plus - loss_minus) / (2 * h)
            gradients.append(grad)
            
        # Update params
        for i in range(n_params):
            params[i] -= learning_rate * gradients[i]
            
    return params
