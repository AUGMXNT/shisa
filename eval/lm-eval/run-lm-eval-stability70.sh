benchmark() {
    LOG_DIR=$(basename "$1")
    mkdir -p "$LOG_DIR"
    echo "$1"
    
    BATCH_SIZE=8
    
    # ARC: 25-shot, arc-challenge (acc_norm) <-- arc_challenge
    time python lm-evaluation-harness-b281b0921b636bc36ad05c0b0b0763bd6dd43463/main.py --model=hf-causal-experimental --model_args="pretrained=$1,use_accelerate=True" --tasks=arc_challenge --num_fewshot=25 --batch_size=$BATCH_SIZE --output_base_path="$LOG_DIR"

    # HellaSwag: 10-shot, hellaswag (acc_norm)
    time python lm-evaluation-harness-b281b0921b636bc36ad05c0b0b0763bd6dd43463/main.py --model=hf-causal-experimental --model_args="pretrained=$1,use_accelerate=True" --tasks=hellaswag --num_fewshot=10 --batch_size=$BATCH_SIZE --output_base_path="$LOG_DIR"

    # TruthfulQA: 0-shot, truthfulqa-mc (mc2)
    time python lm-evaluation-harness-b281b0921b636bc36ad05c0b0b0763bd6dd43463/main.py --model=hf-causal-experimental --model_args="pretrained=$1,use_accelerate=True" --tasks=truthfulqa_mc --num_fewshot=0 --batch_size=$BATCH_SIZE --output_base_path="$LOG_DIR"

    # MMLU: 5-shot, hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions (average of all the results acc)
    time python lm-evaluation-harness-b281b0921b636bc36ad05c0b0b0763bd6dd43463/main.py --model=hf-causal-experimental --model_args="pretrained=$1,use_accelerate=True" --tasks=hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions --num_fewshot=5 --batch_size=$BATCH_SIZE --output_base_path="$LOG_DIR"

    # Winogrande: 5-shot, winogrande (acc)
    time python lm-evaluation-harness-b281b0921b636bc36ad05c0b0b0763bd6dd43463/main.py --model=hf-causal-experimental --model_args="pretrained=$1,use_accelerate=True" --tasks=winogrande --num_fewshot=5 --batch_size=$BATCH_SIZE --output_base_path="$LOG_DIR"

    # GSM8k: 5-shot, gsm8k (acc)
    time python lm-evaluation-harness-b281b0921b636bc36ad05c0b0b0763bd6dd43463/main.py --model=hf-causal-experimental --model_args="pretrained=$1,use_accelerate=True" --tasks=gsm8k --num_fewshot=5 --batch_size=$BATCH_SIZE --output_base_path="$LOG_DIR"

    # DROP: 3-shot, drop (f1)
    time python lm-evaluation-harness-b281b0921b636bc36ad05c0b0b0763bd6dd43463/main.py --model=hf-causal-experimental --model_args="pretrained=$1,use_accelerate=True" --tasks=drop --num_fewshot=3 --batch_size=$BATCH_SIZE --output_base_path="$LOG_DIR"

}

benchmark /mnt/data/models/stabilityai_japanese-stablelm-base-beta-70b 
benchmark /mnt/data/models/stabilityai_japanese-stablelm-instruct-beta-70b