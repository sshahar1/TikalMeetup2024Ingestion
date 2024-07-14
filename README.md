# Ingestion project
This project includes 2 scripts:
1. [fetch_data.py](fetch_data.py) to query data from PredictHQ - Will output a file called Output.json
2. [process_file.py](process_file.py) to prepare embeddings for the data - Will output a file called for_collection.json

To import the data to a Couchbase collection:
1. Install cbimoprt
2. Run cbimport:
```bash
./cbimport json --cluster couchbases://<yourcluster> --username <your user> --password <your password> --bucket <bucket> --scope-collection-exp "<scope>.<collection>" --dataset for_collection.json --generate-key ‘%id%’ --cacert <path to couchbase certificate> --format lines
```