from boto3 import client
from sagemaker.tensorflow.model import TensorFlowModel
import tarfile
from tensorflow import __version__

bucket = "lia-model"
endpoint_name = "lia-model-endpoint"
role = "arn:aws:iam::309100493331:role/LIA_SageMaker_Role"

# Create a tarball of the saved model
with tarfile.open("model.tar.gz", mode="w:gz") as f:
  f.add("model", arcname="model")

# Upload the tarball to S3
s3 = client("s3")
s3.upload_file("model.tar.gz", bucket, "model.tar.gz")

# Create a SageMaker Model object
model = TensorFlowModel(
  framework_version=__version__,
  model_data=f"s3://{bucket}/model.tar.gz",
  role=role
)

model.deploy(
  endpoint_name=endpoint_name,
  initial_instance_count=1,
  instance_type="ml.t2.medium"
)
