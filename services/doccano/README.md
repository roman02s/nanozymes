# Doccano

doccano is an open-source text annotation tool for humans. It provides annotation features for text classification, sequence labeling, and sequence to sequence tasks. You can create labeled data for sentiment analysis, named entity recognition, text summarization, and so on. Just create a project, upload data, and start annotating. You can build a dataset in hours.

link - https://github.com/doccano/doccano

command for run service

```bash
docker pull doccano/doccano
docker container create --name doccano -e "ADMIN_USERNAME=nanozyme_admin" -e "ADMIN_EMAIL=admin@example.com" -e "ADMIN_PASSWORD=nanozyme_password" -v doccano-db:/data -p 9000:8000 doccano/doccano
docker container start doccano
```
