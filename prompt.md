<!-- Write your task prompt here -->
<!-- This is what the agent will see as its instructions -->

Do not install any additional dependencies or use online connectivity while working on this task. You must work offline without internet access.

# Task
I would like to create new two modules. evaluate_tensorflow.py and evaluate_pyTorch.py that load a saved model and output a JSON report
in the following folder structure json/(pyTorch or tensorflow)/<modelname>.json. The module evaluate_tensorflow will use tensorflow and the module evaluate_pyTorch will use pyTorch. The report must include  filename, predicted class, confidence score, and ground truth label. Predictions must use the same preprocessing pipeline as training. Ground truth labels should be inferred from the subdirectory name. The module should expose a run_evaluation(model_path, test_dirs, train_dir threshold=0.5) function and also be runnable as python evaluate.py --model bike_classifier.keras --test-dir images/mountain --test-dir images/road  --train-dir /images/trainset. The arguments in the example should serve as defaults. If no keras file is found with the model name argument then the application will use the code in train.py to create a keras file with that name using the train-dir argument. The foldernames in the training set will be used as ground truth labels.
## Context

The codebase currrently uses tensorflow but I would like to upgrade it to optionally use PyTorch
## Requirements

[List specific requirements]
Both tensorflow and PyTorch versions of the application will work