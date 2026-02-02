# Workspace root (optional) - Set this if you are using eexp_egnine as a service.
# Point this to the root directory where all user workspaces will be created.
WORKSPACE_ROOT = "/exp_engine/workspace"

# Main folders (mandatory) - paths are just examples
TASK_LIBRARY_PATH = 'tasks'
EXPERIMENT_LIBRARY_PATH = 'experiments'
# The ones below has to be a path relative to the script that invokes the client.run()
DATASET_LIBRARY_RELATIVE_PATH = 'datasets'
PYTHON_DEPENDENCIES_RELATIVE_PATH = 'dependencies'
# Reference here the Python file that holds the functions used in the evaluation of conditions.
# Path is relative to the script that invokes the client.run()
PYTHON_CONDITIONS = 'tasks/experiment_conditions'

# The Python file that holds any functions used for filtering or generating configurations for spaces
PYTHON_CONFIGURATIONS = 'tasks/experiment_configurations'
WORKFLOW_LIBRARY_PATH = 'workflows'
# number of workflows that the engine can run in parallel per node at any given moment (if omitted, value is 1)
MAX_WORKFLOWS_IN_PARALLEL_PER_NODE = 1

EXECUTIONWARE = "PROACTIVE" # other options: "LOCAL" & "KUBEFLOW"
# Proactive details (only needed if EXECUTIONWARE = "PROACTIVE" above)
PROACTIVE_URL = "http://146.124.106.171:8880"
# EXECUTIONWARE = "LOCAL"
# PROACTIVE_URL = "http://146.124.106.171:8880"
PROACTIVE_USERNAME="_______"
PROACTIVE_PASSWORD="_______"
# You need to specify the path to the Python version you want to explicitly use (ask ICOM)
PROACTIVE_PYTHON_VERSIONS = {"3.8": "/usr/bin/python3.8", "3.9": "/usr/bin/python3.9"}

KUBEFLOW_URL = ""
KUBEFLOW_USERNAME = ""
KUBEFLOW_PASSWORD = ""
KUBEFLOW_MINIO_ENDPOINT = ""
KUBEFLOW_MINIO_USERNAME = ""
KUBEFLOW_MINIO_PASSWORD = ""

# Data abstraction credentials (mandatory - ask ICOM)
DATA_ABSTRACTION_BASE_URL = "https://api.dal.extremexp-icom.intracom-telecom.com/api"
DATA_ABSTRACTION_ACCESS_TOKEN = '__________'

# possible values: DDM (Decentralized Data Management), LOCAL
DATASET_MANAGEMENT = "LOCAL"
DDM_URL = "https://ddm.extremexp-icom.intracom-telecom.com"
PORTAL_USERNAME = "_________"
PORTAL_PASSWORD = "_________"

# logging configuration, optional; if not set, all loggers have INFO level
LOGGING_CONFIG = {
    'version': 1,
    'loggers': {
        'eexp_engine.functions': {
            'level': 'DEBUG'
        },
        'eexp_engine.functions.parsing': {
            'level': 'DEBUG',
        },
        'eexp_engine.functions.execution': {
            'level': 'DEBUG',
        },
        'eexp_engine.data_abstraction_layer': {
            'level': 'DEBUG'
        },
        'eexp_engine.models': {
            'level': 'DEBUG'
        },
        'eexp_engine.proactive_executionware': {
            'level': 'DEBUG'
        }
    }
}
