import numpy as np
from scipy import stats 

def CellXEstimate(cell):
    if cell is None:
        return (0.0001, 0.0001, 0.0001)
    else:
        if cell.shape[0] == 1:
            x = (cell['magnitude'][0], cell['magnitude'][0], cell['magnitude'][0])
        else:
            x = (cell['magnitude'].min(), cell['magnitude'].median(), cell['magnitude'].max())
    return x


def CellXEstimateVector(cells):
    DoVector = np.vectorize(CellXEstimate)
    min_m, median_m, max_m = DoVector(cells)
    X_list = zip(min_m, median_m, max_m)
    return X_list


def CellYEstimate(cell):
    if cell is None:
        return 0.01
    else:
        return cell.shape[0]


CellYEstimateVector = np.vectorize(CellYEstimate)

# hardcode beta variance
def MuEstimation(y, X):
    log_y = np.log(y)
    sigma_inv_y = np.diag(y)
    sigma_beta_p= np.diag(np.ones(3) * 1.0)
    sigma_beta = np.linalg.inv((X.T.dot(sigma_inv_y).dot(X) + np.linalg.inv(sigma_beta_p)))
    mu_beta = sigma_beta.dot(X.T).dot(sigma_inv_y).dot(log_y)
    return sigma_beta, mu_beta


def PriorDistributionEstimation(Cells):
    X = np.asarray(CellXEstimateVector(Cells)) # observations * regressors matrix
    y = CellYEstimateVector(Cells)
    sigma_beta, mu_beta = MuEstimation(y, X)
    
    return sigma_beta, mu_beta


def PosteriorDistributionUpdate(sigma_beta, mu_beta, x_new, y_new):
    sigma_beta_inv = np.linalg.inv(sigma_beta)
    sigma_beta_new = np.linalg.inv(sigma_beta_inv + x_new.T.dot(x_new) * y_new)
    mu_beta_new = sigma_beta_new.dot(sigma_beta_inv.dot(mu_beta) + x_new * y_new * np.log(y_new))
    
    return sigma_beta_new, mu_beta_new


def BetaStatisticWindowEstimation(t_matrix_train, t_matrix_test, T):
    prior_sigma_beta, prior_mu_beta = PriorDistributionEstimation(t_matrix_train)
    params_beta = {'mu': np.empty(t_matrix_test.shape[0], dtype=object),\
                   'sigma': np.empty(t_matrix_test.shape[0], dtype=object)}
    
    for i in range(t_matrix_test.shape[0]):
        x_new = np.asarray(CellXEstimate(t_matrix_test[i]))
        y_new = CellYEstimate(t_matrix_test[i])
        if i == 0:
            params_beta['sigma'][i], params_beta['mu'][i] = \
            PosteriorDistributionUpdate(prior_sigma_beta, prior_mu_beta, x_new, y_new)
        else:
            if i % T == 0:
                prior_sigma_beta, prior_mu_beta = PriorDistributionEstimation(t_matrix_test[i-T:i])
            else:
                prior_sigma_beta, prior_mu_beta = params_beta['sigma'][i-1], params_beta['mu'][i-1]
            
            params_beta['sigma'][i], params_beta['mu'][i] = \
            PosteriorDistributionUpdate(prior_sigma_beta, prior_mu_beta, x_new, y_new)

    return params_beta


def NormalCdfTail(point, mu, cov):
    diag_elements = np.array([cov[i][i] for i in range(cov.shape[0])])  
    sd = np.sqrt(np.max(diag_elements)) * 5.0
    infty_approx = point - sd
    p, i = stats.mvn.mvnun(infty_approx, point, mu, cov)
    return p


def ProbCurrentMuInPast(past_sigma, past_mu, current_sigma, current_mu):
    prob = NormalCdfTail(current_mu, past_mu, past_sigma)
    return min(prob, 1. - prob)
    

ProbCurrentMuInPastVector = np.vectorize(ProbCurrentMuInPast)


def KLdivergenceNormals(one_sigma, one_mu, zero_sigma, zero_mu):
    inv_sigma_one = np.linalg.inv(one_sigma)
    kl = np.trace(inv_sigma_one.dot(zero_sigma))
    kl = kl + (one_mu - zero_mu).dot(inv_sigma_one).dot(one_mu - zero_mu) - one_sigma.shape[0]
    kl = kl + np.log(np.linalg.det(one_sigma) / np.linalg.det(zero_sigma))
    return kl


KLdivergenceNormalsVector = np.vectorize(KLdivergenceNormals)


def LogOddsRatio(past_sigma, past_mu, current_sigma, current_mu):
    prob_curret_measure = stats.multivariate_normal.logpdf(current_mu, current_mu, current_sigma)
    prob_past_measure = stats.multivariate_normal.logpdf(current_mu, past_mu, past_sigma)
    return prob_curret_measure / prob_past_measure


LogOddsRationVector = np.vectorize(LogOddsRatio)


def LambdaEstimation(sigma_beta, mu_beta, x_new):
    mu_lambda = np.dot(x_new, mu_beta)
    sigma_lambda = (x_new).dot(sigma_beta).dot(x_new)
    return (sigma_lambda, mu_lambda)

def LambdaVectorEstimation(params_beta, x):
    lambda_params = {'mu': np.full(params_beta['mu'].shape[0] - 1, 0.0), \
                    'sigma': np.full(params_beta['mu'].shape[0] - 1, 0.0)} 
    
    for i in range(params_beta['mu'].shape[0] - 1):
        lambda_params['sigma'][i], lambda_params['mu'][i] = LambdaEstimation(params_beta['sigma'][i], \
                                                                             params_beta['mu'][i], x[i+1])
    return lambda_params


def ExpectationPredictionY(sigma_lambda, mu_lambda):
    r = 1 / (sigma_lambda + 0.00001)
    p = np.exp(mu_lambda) / (np.exp(mu_lambda) + 1 / sigma_lambda)
    return r * p / (1 - p)


def ModePredictionY(sigma_lambda, mu_lambda):
    r = 1 / sigma_lambda
    p = np.exp(mu_lambda) / (np.exp(mu_lambda) + 1 / sigma_lambda)
    mode =  p * (r - 1) / (1 - p) if r > 1 else 0
    return mode

ExpectationPredictionYVector = np.vectorize(ExpectationPredictionY)


ModePredictionYVector = np.vectorize(ModePredictionY)


def MakeDictStatistics(T, t_matrix_train, t_matrix_test):
    params = BetaStatisticWindowEstimation(t_matrix_train, t_matrix_test, T)
    current_sigma, current_mu = params['sigma'][1:], params['mu'][1:]
    past_sigma, past_mu = params['sigma'][:-1], params['mu'][:-1]
    p_value = ProbCurrentMuInPastVector(past_sigma, past_mu, current_sigma, current_mu)
    KLFuturePast = KLdivergenceNormalsVector(past_sigma, past_mu, current_sigma, current_mu)
    KLPastFuture = KLdivergenceNormalsVector(current_sigma, current_mu, past_sigma, past_mu)
    log_odds = LogOddsRationVector(past_sigma, past_mu, current_sigma, current_mu)
    x = np.asarray(CellXEstimateVector(t_matrix_test))
    lambda_params = LambdaVectorEstimation(params, x)
    mode_prediction = ModePredictionYVector(lambda_params['sigma'], lambda_params['mu'])
    mean_prediction = ExpectationPredictionYVector(lambda_params['sigma'], lambda_params['mu'])
    
    beta_mu_min = [i[0] for i in  params['mu'][1:]]
    beta_mu_med = [i[1] for i in  params['mu'][1:]]
    beta_mu_max = [i[2] for i in  params['mu'][1:]]
    
    result = {'beta_mu_min': beta_mu_min, 'beta_mu_med': beta_mu_med, 'beta_mu_max': beta_mu_max, 'p_value_beta': p_value,\
              'KLFutrePastBeta': KLFuturePast, 'KLPastFutureBeta': KLPastFuture, 'diff': KLFuturePast - KLPastFuture,
             'LogOdssBeta': log_odds, 'lambda_mu': lambda_params['mu'], 'lambda_sigma': lambda_params['sigma'],\
              'mean_pr_y': mean_prediction, 'mode_pr_y': mode_prediction}
    return result
