# Recommender System on GCP with Kubernetes

This project demonstrates a scalable pipeline for building, training, and deploying a neural collaborative filtering (NCF) model using **Google Cloud Platform (GCP)** and **Kubernetes**. The pipeline includes preprocessing data with Spark, training the model, and deploying it in a containerised environment.

----

## Features
* Data Preprocessing: Utilizes **Apache Spark** to clean and transform raw data stored in GCS.
* Model Training: Trains an NCF model using TensorFlow and pushes the trained model to GCS.
* Containerised Deployment: The entire pipeline runs in Docker containers orchestrated by Kubernetes.
* Scalability: Kubernetes ensures scalability and fault tolerance for model training workloads.
* Environment Management: Leverages GCP services and Kubernetes secrets for secure configuration.

----

## Project Structure

```.
├── Data/                        # Local data directory (ignored by Git)
├── dbt_project/                 # DBT configurations and models for data transformation
│   ├── models/                  # SQL models for cleaned data
│   ├── tests/                   # Testing configurations
│   ├── dbt_project.yml          # DBT project configuration
├── scripts/                     # Python scripts for preprocessing and training
│   ├── preprocess_data.py       # Preprocesses data with Spark
│   ├── train_ncf.py             # Trains the NCF model
├── Dockerfile                   # Docker configuration for model training
├── recommender-deployment.yaml  # Kubernetes deployment configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

----

## Requirements

* Python 3.10 or higher
* Docker
* Google Cloud SDK
* Kubernetes CLI (`kubectl`)

----

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/recommender-system-gcp-k8s.git

cd recommender-system-gcp-k8s
```

2. Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Build and Push Docker Image
Ensure you have authenticated with GCP and set up Artifact Registry:

```bash
docker build -t gcr.io/your-project-id/recommender-training:latest .
docker push gcr.io/your-project-id/recommender-training:latest
```

4. Deploy to Kubernetes

```bash
kubectl apply -f recommender-deployment.yaml
kubectl get pods
```

---

## Workflow

1. Data Preprocessing:
* Raw data from GCS is transformed using a Spark job.
* The transformed data is saved back to GCS.

2.Model Training:
* The NCF model is trained using TensorFlow.
* The trained model is saved to GCS.

3. Deployment:
* The training and preprocessing scripts are containerized using Docker.
* Kubernetes orchestrates the pipeline for scalability and fault tolerance.

----

## Key Configurations

### Environment Variables
The following environment variables are used in the deployment:

* `GOOGLE_APPLICATION_CREDENTIALS`: Path to the service account key for GCP authentication.
* `GOOGLE_CLOUD_PROJECT`: GCP project ID.

----

## Future Enhancements

* Integrate CI/CD with GitHub Actions.
* Add monitoring and logging for Kubernetes pods.
* Explore distributed training with TensorFlow on Kubernetes.

----

## Licence

This project is licensed under the MIT License. See the LICENSE file for details.

----

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to get started.

