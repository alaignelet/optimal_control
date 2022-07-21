# linear example
import ray
import pandas as pd
import numpy as np
import copy
from pde import NeuralNet, LinearQuadraticRegulator2D, NonLinear, LinearQuadraticRegulator10D


@ray.remote
def experimentTrainingPhase(gammaData, gammaGradient, gammaResidual, interiorPointCount, dataPointCount, network_config, training_config):
    
    # network creation
    layers = network_config['layers']
    quadraticForm = network_config['quadraticForm']
    useAdditionalModel = network_config['useAdditionalModel']
    
    network = NeuralNet( 
        layers=layers, 
        quadraticForm=quadraticForm, 
        useAdditionalModel=useAdditionalModel
    )
    
    # pde creation
    gamma = {'data': gammaData, 'gradient': gammaGradient, 'residual': gammaResidual}
    pde = LinearQuadraticRegulator10D( network=network, gamma=gamma )

    
    lrs = training_config['lrs']
    iterations = training_config['iterations']

    resu = pde.train(
        interiorPointCount=interiorPointCount,
        dataPointCount=dataPointCount,
        lrs=lrs,
        iterations=iterations
    )

    info_dict = {}
    info_dict['pde'] = pde
    info_dict['mse'] = resu['mse']
    info_dict['loss'] = resu['loss']
    info_dict['config'] = {
        'gamma': gamma,
        'interiorPointCount': interiorPointCount,
        'dataPointCount': dataPointCount,
        'lrs': lrs,
        'iterations': iterations,
        'layers': layers,
        'quadraticForm': quadraticForm,
        'useAdditionalModel': useAdditionalModel
    }
    
    return info_dict



@ray.remote
def experimentTwoStepsLearning(interiorPointCount, dataPointCount, network_config, training_config):
    
    # network creation
    layers = network_config['layers']
    quadraticForm = network_config['quadraticForm']
    useAdditionalModel = network_config['useAdditionalModel']
    
    network = NeuralNet( 
        layers=layers, 
        quadraticForm=quadraticForm, 
        useAdditionalModel=useAdditionalModel
    )
    
    # pde creation
    gamma_data = {'data': 1., 'gradient': 1., 'residual': 0.}
    pde = NonLinear( network=network, gamma=gamma_data )

    lrs_data = training_config['lrs_data']
    iterations_data = training_config['iterations_data']

    resu_data = pde.train(
        interiorPointCount=interiorPointCount,
        dataPointCount=dataPointCount,
        lrs=lrs_data,
        iterations=iterations_data
    )

    saved_weights = copy.deepcopy( pde.network.state_dict() )

    # second training
    gamma_residual = {'data': 0., 'gradient': 0., 'residual': 1.}
    pde = NonLinear( network=network, gamma=gamma_residual, correctShift=True )

    pde.network.load_state_dict( saved_weights )

    lrs_residual = training_config['lrs_residual']
    iterations_residual = training_config['iterations_residual']

    resu_residual = pde.train(
        interiorPointCount=interiorPointCount,
        dataPointCount=dataPointCount,
        lrs=lrs_residual,
        iterations=iterations_residual
    )

    info_dict = {}
    info_dict['pde'] = pde
    info_dict['mse_data'] = resu_data['mse']
    info_dict['mse_residual'] = resu_residual['mse']
    info_dict['loss_data'] = resu_data['loss']
    info_dict['loss_residual'] = resu_residual['loss']
    info_dict['config'] = {
        'gamma_data': gamma_data,
        'gamma_residual': gamma_residual,
        'interiorPointCount': interiorPointCount,
        'dataPointCount': dataPointCount,
        'lrs_data': lrs_data,
        'lrs_residual': lrs_residual,
        'iterations_data': iterations_data,
        'iterations_residual': iterations_residual,
        'layers': layers,
        'quadraticForm': quadraticForm,
        'useAdditionalModel': useAdditionalModel
    }
    
    return info_dict



def launchClassicExperiment():
    # configure the network
    layers = [10, 100, 100, 100, 1]
    quadraticForm = False
    useAdditionalModel = False

    network_config = {
        'layers': layers,
        'quadraticForm': quadraticForm,
        'useAdditionalModel': useAdditionalModel
    }

    # configure the training
    # lrs = [1e-2, 1e-3, 1e-4]
    # iterations = [1000, 2000, 2000]

    # lrs = [1e-2, 5e-3, 1e-4, 5e-5]
    # iterations = [1000, 2000, 4000, 8000]
    
    # lrs = [1e-3, 1e-4, 1e-5]
    # iterations = [1000, 2000, 2000]

    lrs = [1e-3, 5e-4, 1e-4, 5e-5]
    iterations = [1000, 2000, 4000, 8000]

    training_config = {
        'lrs': lrs,
        'iterations': iterations
    }

    dataPointCount = 500
    interiorPointCount = 500

    nexp = 10
    # params = [[0, 0, 1, interiorPointCount, dataPointCount]] * nexp
    params = [[1, 0, 0, interiorPointCount, dataPointCount]] * nexp
    params += [[1, 1, 0, interiorPointCount, dataPointCount]] * nexp
    params += [[1, 1, 0.1, interiorPointCount, dataPointCount]] * nexp

    # params = [[1, 1, 0.1, interiorPointCount, dataPointCount]] * nexp
    
    # params = [[0, 1, 1, interiorPointCount, dataPointCount]] * nexp

    resu = []

    for gammaData, gammaGradient, gammaResidual, interiorPointCount, dataPointCount in params:
        resu_exp = experimentTrainingPhase.remote(
            gammaData=gammaData,
            gammaGradient=gammaGradient,
            gammaResidual=gammaResidual,
            interiorPointCount=interiorPointCount,
            dataPointCount=dataPointCount,
            network_config=network_config,
            training_config=training_config
        )
        resu.append(resu_exp)

    resu = ray.get(resu)
    resu = pd.DataFrame(resu)
    resu.to_pickle("experiments_linear_10d.csv")


def launchTwoStepsLearningExperiment():
    # configure the network
    layers = [2, 20, 20, 20, 1]
    quadraticForm = False
    useAdditionalModel = False

    network_config = {
        'layers': layers,
        'quadraticForm': quadraticForm,
        'useAdditionalModel': useAdditionalModel
    }

    # configure the training for data and residual
    lrs_data = [1e-2, 1e-3]
    iterations_data = [1000, 2000]
    lrs_residual = [1e-2, 1e-3, 1e-4]
    iterations_residual = [2000, 4000, 9000]

    training_config = {
        'lrs_data': lrs_data,
        'iterations_data': iterations_data,
        'lrs_residual': lrs_residual,
        'iterations_residual': iterations_residual,
    }

    resu = []

    dataPointCount = 50
    interiorPointCount = 100

    params = [[interiorPointCount, dataPointCount]] * 10

    for interiorPointCount, dataPointCount in params:
        resu_exp = experimentTwoStepsLearning.remote(
            interiorPointCount=interiorPointCount,
            dataPointCount=dataPointCount,
            network_config=network_config,
            training_config=training_config,
        )
        resu.append(resu_exp)

    resu = ray.get(resu)
    resu = pd.DataFrame(resu)
    resu.to_pickle("experiments_non_linear_two_steps.csv")



if __name__ == "__main__":
    ray.shutdown()
    ray.init(num_cpus=6)

    # launchTwoStepsLearningExperiment()
    launchClassicExperiment()
