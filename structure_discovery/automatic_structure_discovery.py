import pyGPs
import numpy as np
import pandas as pd
import pylab as plt
import pyGPs
import copy

def load_mauna_loa_ds():
    df = pd.read_csv('../test/maunaloa.co2.txt', sep='\t', index_col=0)
    del df['Annual Average']
    return df.stack()


def extract_data_from_df(df):
    y = np.asarray(df)
    x = np.arange(len(y))
    
    return x, y


def get_standard_kernels():    
    Lin = pyGPs.cov.Linear
    SE = pyGPs.cov.RBF
    Per = pyGPs.cov.Periodic
    RQ = pyGPs.cov.RQ
    
    return Lin, SE, Per, RQ


def extract_kernel_composition(kernel):
    if type(kernel) == pyGPs.Core.cov.ProductOfKernel:
        return [pyGPs.Core.cov.ProductOfKernel, 
                extract_kernel_composition(kernel.cov1),
                extract_kernel_composition(kernel.cov2)]
    elif type(kernel) == pyGPs.Core.cov.SumOfKernel:
        return [pyGPs.Core.cov.SumOfKernel, 
                extract_kernel_composition(kernel.cov1),
                extract_kernel_composition(kernel.cov2)]
    else:
        return type(kernel)


def generate_kernel_from_composition(comp):
    if type(comp[1]) == list:
        a1 = generate_kernel_from_composition(comp[1])
    else:
        a1 = comp[1]
        
    if type(comp[2]) == list:
        a2 = generate_kernel_from_composition(comp[2])
    else:
        a2 = comp[2]
        
    return comp[0](a1, a2)
    
    
def generate_model_from_paper(df):
    x, y = extract_data_from_df(df)
    Lin, SE, Per, RQ = get_standard_kernels()    
    k = Lin() + SE() * SE() * (Per() + RQ())
        
    model = pyGPs.GPR()
    model.setPrior(kernel=k)
    model.getPosterior(x, y)
    model.setOptimizer("Minimize", num_restarts=1)
    model.optimize()
    
    return model


def train_model(model, kernel, df):
    x, y = extract_data_from_df(df)
    
    model = pyGPs.GPR()
    model.setPrior(kernel=kernel)
    model.getPosterior(x, y)
    model.setOptimizer("Minimize", num_restarts=1)
    model.optimize()
    
    return model

def evalutate_model_fitness(model, df):
    """ according to p. 37
    BIC(M) = log p(D | M) − 0.5 |M| log N
    
    with p(D|M) being  marginal likelihood
         |M|           number of hyper parameters
         N             number of datapoints
    
    see also:
        https://github.com/marionmari/pyGPs/blob/master/pyGPs/Core/gp.py#L73
    """
    return model.nlZ - 0.5 * len(model.covfunc.hyp) * np.log(model.x.shape[0])


def fit_new_kernel(cur_model, new_kernel, kernel_func, df):
    if cur_model is None:        
        cur_model = pyGPs.GPR()
        cur_model.setOptimizer("Minimize")
        tmp_kernel = new_kernel()
    else:
        tmp_comp = [kernel_func,
                    new_kernel(),
                    cur_model.covfunc]
        tmp_kernel = generate_kernel_from_composition(tmp_comp)
    
    new_model = train_model(cur_model, 
                            tmp_kernel, 
                            df)
    
    fitness = evalutate_model_fitness(new_model, df)
    
    return new_model, fitness


def discover_model_step(df, cur_model=None, kernels='all'):    
    if kernels == 'all':
        kernels = get_standard_kernels()
        
    models = {}
    fitness = []
    
    for k in kernels:
        for kernel_func in [pyGPs.Core.cov.ProductOfKernel,
                            pyGPs.Core.cov.SumOfKernel]:
            
            new_model, new_fit = fit_new_kernel(copy.deepcopy(cur_model),
                                                k,
                                                kernel_func,
                                                df)
            fitness += [[new_model, new_fit]]
            
    best_model_pos = np.argmin(zip(*fitness)[1])
    
    return fitness[best_model_pos]


def discover_model(df, kernels='all'):
    model = discover_model_step(df)
    model2 = discover_model_step(df, model[0])
    
    return model, model2


def structure_discovery(df, kernels=None):
    if kernels is None:
        kernels = [pyGPs.cov.Linear, pyGPs.cov.Periodic, pyGPs.cov.RBF]
        
    model = pyGPs.GPR()
    model.setOptimizer("Minimize", num_restarts=30)


if __name__ == "__main__":
    df = load_mauna_loa_ds()

    model = discover_model_step(df)

    print m.covfunc
    print m.covfunc.hyp