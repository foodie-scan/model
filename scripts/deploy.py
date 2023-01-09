from boto3 import client
from sagemaker.tensorflow import TensorFlowModel
import tarfile

bucket = "lia-model"
endpoint_name = "lia-model-endpoint"
saved_model = "model"
key = f"{saved_model}.tar.gz"
role = "arn:aws:iam::309100493331:role/LIA_SageMaker_Role"

# Create a tarball of the saved model
with tarfile.open(key, mode="w:gz") as f:
  f.add(saved_model, arcname=saved_model)

# Upload the tarball to S3
s3 = client("s3")
s3.upload_file(key, bucket, key)

# Create a SageMaker Model object
model = TensorFlowModel(
  framework_version="2.8",
  model_data=f"s3://{bucket}/{key}",
  role=role
)

model.deploy(
  endpoint_name=endpoint_name,
  initial_instance_count=1,
  instance_type="ml.t2.medium"
)
