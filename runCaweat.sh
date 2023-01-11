#!/bin/bash

RESULTDIR=`pwd`"/results"

# Verifying that folder results exists
if [[ ! -d $RESULTDIR ]]
then
    echo "FATAL. Directory" $RESULTDIR  "does not exist."
    exit 1
fi


# Starting the process
for similarity_type in "cosine"; do
    # List the languages you want to consider here
    for language in "es_ES1" "es_ES2" "es_ES3" "es_EC1" "es_EC2" "es_CO1" "es_CO2" "es_MX1" "en_US1" "en_US2" "en_US3" "en_US4" "en_US5";  do
        #  Considers both WEAT1 and WEAT2
        for test_number in 1 2; do
            python3 caweat.py \
                --test_number $test_number \
                --permutation_number 1000000 \
                --bootstrap_number 5000 \
                --output_file $RESULTDIR/ca_bert0_${language}_${similarity_type}_${test_number}_uncased.res \
                --lower True \
                --lang $language \
                --embeddings ./emb/bertEMB.layer0${language:0:2}.vec\
                --similarity_type $similarity_type |& tee $RESULTDIR/origs/sigmas/en/ca_${model}_${language}_${similarity_type}_${test_number}_uncased.out
        done
    done
done
