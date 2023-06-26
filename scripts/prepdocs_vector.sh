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

echo 'Creating python virtual environment "scripts/.venv"'
#python -m venv scripts/.venv

echo 'Installing dependencies from "requirements.txt" into virtual environment'
#./scripts/.venv/bin/python -m pip install -r scripts/requirements.txt

echo 'Running "prepdocs_vector.py"'
./scripts/.venv/bin/python ./scripts/prepdocs_vector.py './data/hcc_docs_event/hcc_card_event_00009.pdf' --storageaccount "$AZURE_STORAGE_ACCOUNT" --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --index "hcc-poc-index-vector" --category "Card Event" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" -v --skipblobs
#./scripts/.venv/bin/python ./scripts/prepdocs_vector.py './data/hcc_docs_guide/hcc_card_guide_00010.pdf' --storageaccount "$AZURE_STORAGE_ACCOUNT" --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --index "hcc-poc-index-vector" --category "Card Guide" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" -v 

#./scripts/.venv/bin/python ./scripts/prepdocs_vector.py './data/hcc_docs_event/hcc_card_event_0006*' --storageaccount "$AZURE_STORAGE_ACCOUNT" --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --index "hcc-poc-index-vector" --category "Event" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" --removeall
#./scripts/.venv/bin/python ./scripts/prepdocs.py './data/*' --storageaccount "$AZURE_STORAGE_ACCOUNT" --container "$AZURE_STORAGE_CONTAINER" --searchservice "$AZURE_SEARCH_SERVICE" --index "$AZURE_SEARCH_INDEX" --formrecognizerservice "$AZURE_FORMRECOGNIZER_SERVICE" --tenantid "$AZURE_TENANT_ID" -v
