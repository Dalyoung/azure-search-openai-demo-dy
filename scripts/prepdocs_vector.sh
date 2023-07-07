 #!/bin/sh

echo ""
echo "DY-Vector Upload.... Loading azd .env file from current environment"
echo ""

while IFS='=' read -r key value; do
    value=$(echo "$value" | sed 's/^"//' | sed 's/"$//')
    export "$key=$value"
done <<EOF
$(azd env get-values)
EOF

#echo 'Creating python virtual environment "scripts/.venv"'
#python -m venv scripts/.venv

#echo 'Installing dependencies from "requirements.txt" into virtual environment'
#./scripts/.venv/bin/python -m pip install -r scripts/requirements.txt

echo 'Running "prepdocs_vector.py"'

#./scripts/.venv/bin/python ./scripts/prepdocs_vector.py './data/hcc_docs_event/hcc_card_event_00004.pdf' --title '가맹점 업종별 무이자 할부/부분 무이자 할부' --category '이벤트'  --index 'hcc-poc-index2' --storageaccount "$AZURE_STORAGE_ACCOUNT"  --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" -v --skipblobs  
./scripts/.venv/bin/python ./scripts/prepdocs_vector.py './data/hcc_docs_info/hcc_card_info_00001.pdf' --title '현대카드M2 Edition3 (Family)' --category '신용카드 설명서'  --index 'hcc-poc-index2' --storageaccount "$AZURE_STORAGE_ACCOUNT"  --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" -v --skipblobs  
#./scripts/.venv/bin/python ./scripts/prepdocs_vector.py './data/hcc_docs_event/hcc_card_event_00003.pdf' --title '가맹점 업종별 무이자 할부/부분 무이자 할부' --category 'Card Event' --index 'hcc-poc-index2' --storageaccount "$AZURE_STORAGE_ACCOUNT" --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" -v --skipblobs 
#./scripts/.venv/bin/python ./scripts/prepdocs_vector.py './data/hcc_docs_event/hcc_card_event_00001.pdf' --title '대상 카드로 4대 보험료 5만원 이상 납부 시 2~3개월 무이자 할부, 10·12개월 부분 무이자 할부' --category '이벤트'  --index 'hcc-poc-index2' --storageaccount "$AZURE_STORAGE_ACCOUNT"  --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" -v --skipblobs  
#./scripts/.venv/bin/python ./scripts/prepdocs_vector.py './data/hcc_docs_guide/hcc_card_guide_00010.pdf' --storageaccount "$AZURE_STORAGE_ACCOUNT" --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --index "hcc-poc-index-vector" --category "Card Guide" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" -v 

#./scripts/.venv/bin/python ./scripts/prepdocs_vector.py './data/hcc_docs_event/hcc_card_event_0006*' --storageaccount "$AZURE_STORAGE_ACCOUNT" --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --index "hcc-poc-index-vector" --category "Event" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" --removeall
#./scripts/.venv/bin/python ./scripts/prepdocs.py './data/*' --storageaccount "$AZURE_STORAGE_ACCOUNT" --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --index "$AZURE_SEARCH_INDEX" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" -v
