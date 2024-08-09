import mlflow

mlflow.set_tracking_uri("http://localhost:5000")


with mlflow.start_run(run_name="RandomForestClassifier") as run:
    pass

mlflow.end_run()

n_estimators = 100
criterion = "gini"
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
min_weight_fraction_leaf = 0.0
max_features = "sqrt"
max_leaf_nodes = None
min_impurity_decrease = 0.0
bootstrap = True
oob_score = False
n_jobs = None
random_state = None
verbose = 0
warm_start = False
class_weight = None
ccp_alpha = 0.0
max_samples = None
monotonic_cst = None
