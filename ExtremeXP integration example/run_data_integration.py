from eexp_engine import client
import eexp_config
import logging
logging.basicConfig(level=logging.INFO)
# a0cJQZsBbifMolPMI1Nu
if __name__ == "__main__":
    experiment_file = "src/experiments/DataIntegrationExperiment.xxp"
    experiment_name = "DataIntegrationExperiment"

    client.run(experiment_file, experiment_name, eexp_config)
