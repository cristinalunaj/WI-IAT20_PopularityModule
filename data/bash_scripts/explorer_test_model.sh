#!/bin/sh

# Extract predictions from model
# Input arguments:
# TEST = arff file containing the popularity module features and samples to eval
# MODEL_PATH = Path to the model to use (e.g. data/models/popularity_module/CLASIF/th05/RandomForest.model)
# OUTPUT_FILE = Output file to save predictions

TEST="$1"
MODEL_PATH="$2"
OUTPUT_FILE="$3"



printf "\t[TEST]=[%s]\n" "$TEST"
printf "\t[MODEL_PATH]=[%s]\n" "$MODEL_PATH"
printf "\t[OUTPUT_FILE]=[%s]\n" "$OUTPUT_FILE"

java -cp ../../libs/weka/weka.jar weka.classifiers.trees.RandomForest -l "$MODEL_PATH" -T "$TEST" -p 0 -classifications "weka.classifiers.evaluation.output.prediction.CSV -decimals 7" > "$OUTPUT_FILE"

