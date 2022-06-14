###LOWERCASE!!
model='bert0en'
for similarity_type in "cosine"; do
    for language in "es_ES1" "es_ES2" "es_ES3" "es_EC1" "es_EC2" "es_CO1" "es_CO2" "es_MX1" "en_US1" "en_US2" "en_US3" "en_US4" "en_US5";  do
#    for language in "it1" "it2" "it3" "it4" "it5" "it6" "it7" "it9" "it10" "it11" "it12" "it13" "it14" "it15" "it16" "it17" "it18" "it19" "it20" "it21" "it22" "it23" "it24" "it25"; do
        for test_number in 1 2; do
            python3 caweat2Sigma.py \
                --test_number $test_number \
                --permutation_number 1000000 \
                --bootstrap_number 5000 \
                --output_file ./results/origs/sigmas/en/ca_${model}_${language}_${similarity_type}_${test_number}_uncased.res \
                --lower True \
                --lang $language \
                --embeddings ./emb/bertEMB.layer0${language:0:2}.vec\
                --similarity_type $similarity_type |& tee ./results/origs/sigmas/en/ca_${model}_${language}_${similarity_type}_${test_number}_uncased.out
        done
    done
done
#               --embeddings ./emb/cc100_17vecmap.${language:0:2}.vec\
#               --output_file ./results/origs/sigmas/de/ca_${model}_${language}_${similarity_type}_${test_number}_uncased.res \


